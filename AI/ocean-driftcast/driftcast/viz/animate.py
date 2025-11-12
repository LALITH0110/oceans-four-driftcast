# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Renders driftcast simulation outputs into preview or final-cut animations.
- Adds particle trails, density backdrops, lower-third overlays, and credits.
- Relies on safe FFmpeg detection with Matplotlib or imageio fallbacks.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr
from driftcast import logger
import cartopy.crs as ccrs
from matplotlib import animation
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from driftcast.config import SimulationConfig, load_config
from driftcast.post.density import particle_density
from driftcast.sim.runner import run_simulation
from driftcast.viz.ffmpeg import WriterConfig, safe_writer
from driftcast.viz.map import make_basemap
from driftcast.viz.style import apply_style


@dataclass(frozen=True)
class AnimationSettings:
    """Container for animation dial settings."""

    fps: int
    frame_repeat: int
    width_px: int
    height_px: int
    dpi: int = 120
    title_seconds: float = 3.0
    credit_seconds: float = 6.0
    trails_length: int = 8
    show_density: bool = True
    density_alpha: float = 0.35
    codec: str = "libx264"
    bitrate: int = 6000

    @property
    def figsize(self) -> Tuple[float, float]:
        return (self.width_px / self.dpi, self.height_px / self.dpi)


@dataclass
class AnimationScene:
    """Reusable scene description for rendering frames or writing video."""

    fig: plt.Figure
    frames: List[dict]
    init_func: Callable[[], List]
    update_func: Callable[[dict], List]
    context: Dict[str, object]


def preview_settings() -> AnimationSettings:
    return AnimationSettings(
        fps=24,
        frame_repeat=2,
        width_px=1280,
        height_px=720,
        trails_length=6,
        bitrate=4500,
    )


def final_settings() -> AnimationSettings:
    return AnimationSettings(
        fps=24,
        frame_repeat=8,
        width_px=1920,
        height_px=1080,
        trails_length=12,
        bitrate=9000,
    )


def _apply_runtime_days(cfg: SimulationConfig, days: Optional[int]) -> SimulationConfig:
    if not days:
        return cfg
    new_end = cfg.time.start + timedelta(days=days)
    new_time = cfg.time.model_copy(update={"end": new_end})
    return cfg.model_copy(update={"time": new_time})


def _apply_physics_preset(cfg: SimulationConfig, preset: Optional[str]) -> SimulationConfig:
    if not preset:
        return cfg
    physics = cfg.physics
    update: Dict[str, float] = {}
    beach_update: Dict[str, float | None] = {}
    preset_lower = preset.lower()
    if preset_lower == "microplastic_default":
        update = {
            "diffusivity_m2s": 30.0,
            "windage_coeff": 0.002,
            "stokes_coeff": 0.05,
        }
        beach_update = {"probability": 0.02, "resuspension_days": 1.0 / 0.002}
    elif preset_lower == "macro_default":
        update = {
            "diffusivity_m2s": 15.0,
            "windage_coeff": 0.02,
            "stokes_coeff": 0.15,
        }
        beach_update = {"probability": physics.beaching.probability, "resuspension_days": physics.beaching.resuspension_days}
    else:
        return cfg
    new_beaching = physics.beaching.model_copy(update=beach_update) if beach_update else physics.beaching
    new_physics = physics.model_copy(update={**update, "beaching": new_beaching})
    tagged = cfg.model_copy(update={"physics": new_physics})
    return tagged


def _prepare_config(config_path: Path | str, days: Optional[int], preset: Optional[str]) -> SimulationConfig:
    cfg = load_config(config_path)
    cfg = _apply_runtime_days(cfg, days)
    cfg = _apply_physics_preset(cfg, preset)
    return cfg


def _with_ekman(cfg: SimulationConfig, enabled: bool) -> SimulationConfig:
    payload = cfg.model_dump(mode="python")
    physics = payload.setdefault("physics", {})
    ekman = physics.setdefault("ekman", {})
    ekman["enabled"] = enabled
    return SimulationConfig(**payload)


def _frame_seconds(time_values: np.ndarray) -> np.ndarray:
    if time_values.size == 0:
        return np.array([])
    base = time_values.astype("datetime64[ns]")
    seconds = (base - base[0]) / np.timedelta64(1, "s")
    return seconds.astype(float)


def _prepare_longcut_scene(
    config_path: Path | str,
    preset: str,
    duration_minutes: float,
    seed: Optional[int],
) -> Tuple[
    AnimationSettings,
    AnimationScene,
    xr.Dataset,
    Optional[Legend],
    plt.Text,
    plt.Text,
    List[Tuple[float, str]],
    np.ndarray,
    str,
]:
    duration = float(np.clip(duration_minutes, 2.0, 10.0))
    cfg = _prepare_config(config_path, 180, preset)
    dataset = run_simulation(cfg, seed=seed)
    dataset.attrs["preset_name"] = f"longcut_{preset}"
    base_settings = AnimationSettings(
        fps=24,
        frame_repeat=6,
        width_px=1920,
        height_px=1080,
        trails_length=14,
        title_seconds=3.0,
        credit_seconds=6.0,
        show_density=True,
        density_alpha=0.35,
        bitrate=9200,
    )
    data_frames = max(dataset.sizes.get("time", 1), 1)
    usable_seconds = max(duration * 60.0 - base_settings.title_seconds - base_settings.credit_seconds, 60.0)
    frame_repeat = int(round(usable_seconds * base_settings.fps / data_frames))
    frame_repeat = int(np.clip(frame_repeat, 2, 12))
    settings = AnimationSettings(
        fps=base_settings.fps,
        frame_repeat=frame_repeat,
        width_px=base_settings.width_px,
        height_px=base_settings.height_px,
        dpi=base_settings.dpi,
        title_seconds=base_settings.title_seconds,
        credit_seconds=base_settings.credit_seconds,
        trails_length=max(base_settings.trails_length, frame_repeat // 2),
        show_density=base_settings.show_density,
        density_alpha=base_settings.density_alpha,
        codec=base_settings.codec,
        bitrate=base_settings.bitrate,
    )
    scenario = f"{Path(config_path).stem} - long cut"
    scene = create_animation_scene(cfg, dataset, settings, scenario_name=scenario)
    ax = scene.context["ax"]  # type: ignore[index]
    palette = scene.context.get("palette", {})
    legend_handles = [Line2D([], [], marker="o", linestyle="None", markersize=6, color=color) for color in palette.values()]
    legend_labels = [name.replace("_", " ") for name in palette.keys()]
    legend = ax.legend(legend_handles, legend_labels, loc="upper right", frameon=False) if legend_handles else None
    if legend is not None:
        legend.set_alpha(0.0)
    scene_label = ax.text(
        0.02,
        0.9,
        "",
        transform=ax.transAxes,
        color="#f8d568",
        fontsize=18,
        weight="bold",
        bbox=dict(facecolor="#0b1d3a", alpha=0.55, pad=6, edgecolor="none"),
        alpha=0.0,
    )
    watermark = ax.text(
        0.98,
        0.02,
        "Driftcast 2025 Oceans Four",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#f1f0ea",
        alpha=0.35,
        fontsize=11,
    )
    sections = [
        (0.25, "Scene 1 - Atlantic Overview"),
        (0.5, "Scene 2 - Source Mix"),
        (0.75, "Scene 3 - Gyre Convergence"),
        (1.01, "Scene 4 - Beaching Hotspots"),
    ]
    frame_seconds = _frame_seconds(dataset.time.values if "time" in dataset.coords else np.array([]))
    return settings, scene, dataset, legend, scene_label, watermark, sections, frame_seconds, scenario


def _apply_longcut_overlays(
    scene: AnimationScene,
    settings: AnimationSettings,
    dataset: xr.Dataset,
    legend: Optional[Legend],
    scene_label: plt.Text,
    watermark: plt.Text,
    sections: List[Tuple[float, str]],
    frame_seconds: np.ndarray,
    caption_text: Optional[plt.Text] = None,
    captions: Optional[List[Tuple[float, float, str]]] = None,
) -> None:
    data_frames = max(dataset.sizes.get("time", 1), 1)
    total_data_calls = data_frames * settings.frame_repeat
    data_counter = 0
    section_index = 0
    original_update = scene.update_func
    caption_segments = captions or []
    caption_ptr = 0

    def _active_caption(current_time: float) -> str:
        nonlocal caption_ptr
        if not caption_segments:
            return ""
        while caption_ptr < len(caption_segments) and current_time > caption_segments[caption_ptr][1]:
            caption_ptr += 1
        if caption_ptr < len(caption_segments):
            start, end, text = caption_segments[caption_ptr]
            if start <= current_time <= end:
                return text
        return ""

    def update(frame_info: dict):
        nonlocal data_counter, section_index
        artists = original_update(frame_info)
        if frame_info["kind"] == "data":
            data_counter += 1
            progress = data_counter / max(total_data_calls, 1)
            while section_index < len(sections) and progress > sections[section_index][0]:
                section_index += 1
            label = sections[min(section_index, len(sections) - 1)][1]
            scene_label.set_text(label)
            scene_label.set_alpha(0.9)
            if legend is not None:
                if 0.25 <= progress < 0.5:
                    fade = min(1.0, (progress - 0.25) / 0.1)
                    legend.set_alpha(fade)
                else:
                    legend.set_alpha(max(0.0, legend.get_alpha() - 0.05))
            if caption_text is not None:
                idx = min(frame_info["index"], frame_seconds.size - 1) if frame_seconds.size else 0
                caption_line = _active_caption(frame_seconds[idx] if frame_seconds.size else 0.0)
                if caption_line:
                    caption_text.set_text(caption_line)
                    caption_text.set_alpha(0.95)
                else:
                    caption_text.set_alpha(0.0)
            artists.append(scene_label)
            if legend is not None and legend.get_alpha() > 0.0:
                artists.append(legend)
            artists.append(watermark)
            if caption_text is not None and caption_text.get_alpha() > 0.0:
                artists.append(caption_text)
        elif frame_info["kind"] == "title":
            scene_label.set_alpha(0.0)
            if legend is not None:
                legend.set_alpha(0.0)
            if caption_text is not None:
                caption_text.set_alpha(0.0)
        elif frame_info["kind"] == "credits":
            scene_label.set_alpha(0.0)
            if legend is not None:
                legend.set_alpha(0.0)
            if caption_text is not None:
                caption_text.set_alpha(0.0)
            artists.append(watermark)
        return artists

    scene.update_func = update


def _parse_srt_timestamp(timestamp: str) -> float:
    hours, minutes, rest = timestamp.split(":", 2)
    seconds, millis = rest.split(",", 1)
    total = (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )
    return float(total)


def _parse_srt(path: Path) -> List[Tuple[float, float, str]]:
    entries: List[Tuple[float, float, str]] = []
    block: List[str] = []
    for line in path.read_text(encoding="utf8").splitlines() + [""]:
        if line.strip():
            block.append(line.strip())
            continue
        if len(block) >= 2:
            timing = block[1]
            if "-->" in timing:
                start_str, end_str = [part.strip() for part in timing.split("-->")]
                start = _parse_srt_timestamp(start_str)
                end = _parse_srt_timestamp(end_str)
                text = " ".join(block[2:])
                entries.append((start, end, text))
        block = []
    return entries


GYRE_BOX = {"lon_min": -70.0, "lon_max": -30.0, "lat_min": 20.0, "lat_max": 40.0}


def _inside_gyre(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return (
        (lon >= GYRE_BOX["lon_min"])
        & (lon <= GYRE_BOX["lon_max"])
        & (lat >= GYRE_BOX["lat_min"])
        & (lat <= GYRE_BOX["lat_max"])
    )


def _source_palette(sources: Iterable[str]) -> Dict[str, str]:
    palette = [
        "#f8d568",
        "#76b7b2",
        "#f28e2b",
        "#4e79a7",
        "#e15759",
        "#59a14f",
        "#edc949",
        "#af7aa1",
    ]
    colors: Dict[str, str] = {}
    for idx, src in enumerate(sorted(set(sources))):
        colors[src] = palette[idx % len(palette)]
    return colors


def _prepare_frames(dataset: xr.Dataset, settings: AnimationSettings) -> List[dict]:
    frames: List[dict] = []
    frames.extend([{"kind": "title"}] * int(settings.title_seconds * settings.fps))
    for idx in range(dataset.sizes.get("time", 0)):
        frames.extend([{"kind": "data", "index": idx}] * settings.frame_repeat)
    frames.extend([{"kind": "credits"}] * int(settings.credit_seconds * settings.fps))
    return frames


def _lower_third_text(
    scenario_name: str,
    dataset: xr.Dataset,
    frame_idx: int,
    active_count: int,
    config: SimulationConfig,
) -> str:
    time_values = dataset.time.values
    if time_values.size == 0:
        day_label = "Day 0.0"
    else:
        delta_days = (
            (time_values[frame_idx] - time_values[0]) / np.timedelta64(1, "D")
            if frame_idx < len(time_values)
            else 0.0
        )
        day_label = f"Day {delta_days:.1f}"
    gyre_box = getattr(config, "gyre_box", None)
    if gyre_box is not None:
        gyre_text = (
            f" | Gyre box {gyre_box.lat_min:.0f}-{gyre_box.lat_max:.0f}N, "
            f"{abs(gyre_box.lon_max):.0f}-{abs(gyre_box.lon_min):.0f}W"
        )
    else:
        gyre_text = ""
    return (
        f"{scenario_name} | {day_label}{gyre_text}\n"
        f"Particles: {active_count} | Kh={config.physics.diffusivity_m2s:.1f} m^2/s | "
        f"windage={config.physics.windage_coeff:.3f} | stokes={config.physics.stokes_coeff:.3f}"
    )


def _trail_rgba(base_rgba: np.ndarray, age: int, length: int) -> np.ndarray:
    rgba = base_rgba.copy()
    fade = (age + 1) / max(length, 1)
    rgba[-1] = 0.2 * fade
    return rgba


def create_animation_scene(
    config: SimulationConfig,
    dataset: xr.Dataset,
    settings: AnimationSettings,
    scenario_name: str,
) -> AnimationScene:
    """Construct the animation scene used for rendering and testing."""
    apply_style()
    fig, ax = make_basemap(config.domain, figsize=settings.figsize, draw_grid=False)

    heat = None
    if settings.show_density:
        heat = ax.imshow(
            np.zeros((100, 100)),
            extent=(
                config.domain.lon_min,
                config.domain.lon_max,
                config.domain.lat_min,
                config.domain.lat_max,
            ),
            origin="lower",
            cmap="viridis",
            alpha=settings.density_alpha,
            zorder=1,
        )

    scatter = ax.scatter([], [], s=22, alpha=0.9, zorder=3, edgecolors="none")
    trail_scatter = ax.scatter([], [], s=8, alpha=0.0, zorder=2, edgecolors="none")
    lower_text = ax.text(
        0.02,
        0.05,
        "",
        transform=ax.transAxes,
        color="#f1f0ea",
        fontsize=13,
        bbox=dict(facecolor="#0b1d3a", alpha=0.65, pad=6, edgecolor="none"),
    )
    caption_text = ax.text(
        0.5,
        0.12,
        "",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#f1f0ea",
        fontsize=16,
        bbox=dict(facecolor="#0b1d3a", alpha=0.55, pad=6, edgecolor="none"),
        alpha=0.0,
        wrap=True,
    )
    title_text = ax.text(
        0.5,
        0.82,
        scenario_name,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#f1f0ea",
        fontsize=32,
        weight="bold",
    )
    subtitle_text = ax.text(
        0.5,
        0.72,
        "Synthetic North Atlantic Driftcast",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#f1f0ea",
        fontsize=16,
    )
    credit_text = ax.text(
        0.5,
        0.5,
        "Team Oceans Four • Illinois Tech Grainger Computing Innovation Prize",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#f1f0ea",
        fontsize=18,
        alpha=0.0,
        wrap=True,
    )

    frames = _prepare_frames(dataset, settings)
    n_particles = dataset.sizes.get("particle", 0)
    source_coord = dataset.coords.get("source_name")
    palette = _source_palette(
        source_coord.values if source_coord is not None else ["source"]
    )
    source_values = (
        dataset.source_name.values
        if "source_name" in dataset.coords
        else np.array(["source"] * n_particles)
    )
    color_array = np.array([palette.get(src, "#ffffff") for src in source_values])
    rgba_colors = np.array([mcolors.to_rgba(col) for col in color_array])
    color_state: Dict[str, np.ndarray] = {"colors": color_array, "rgba": rgba_colors}
    trail_history: List[Deque[Tuple[float, float]]] = [
        deque(maxlen=settings.trails_length) for _ in range(n_particles)
    ]

    def init():
        if heat is not None:
            heat.set_data(np.zeros_like(heat.get_array()))
        scatter.set_offsets(np.zeros((0, 2)))
        trail_scatter.set_offsets(np.zeros((0, 2)))
        trail_scatter.set_alpha(0.0)
        lower_text.set_text("")
        caption_text.set_alpha(0.0)
        caption_text.set_text("")
        title_text.set_alpha(1.0)
        subtitle_text.set_alpha(1.0)
        credit_text.set_alpha(0.0)
        return (
            [artist for artist in [heat, scatter, trail_scatter, lower_text, title_text, subtitle_text, credit_text, caption_text] if artist is not None]
        )

    def update(frame_info: dict):
        kind = frame_info["kind"]
        if kind == "title":
            title_text.set_alpha(1.0)
            subtitle_text.set_alpha(1.0)
            credit_text.set_alpha(0.0)
            trail_scatter.set_offsets(np.zeros((0, 2)))
            trail_scatter.set_alpha(0.0)
            scatter.set_offsets(np.zeros((0, 2)))
            lower_text.set_text("")
            caption_text.set_alpha(0.0)
            caption_text.set_text("")
            if heat is not None:
                heat.set_data(np.zeros_like(heat.get_array()))
            return init()

        if kind == "credits":
            title_text.set_alpha(0.0)
            subtitle_text.set_alpha(0.0)
            credit_text.set_alpha(1.0)
            lower_text.set_text("Thank you for evaluating Driftcast.")
            caption_text.set_alpha(0.0)
            caption_text.set_text("")
            scatter.set_offsets(np.zeros((0, 2)))
            trail_scatter.set_offsets(np.zeros((0, 2)))
            trail_scatter.set_alpha(0.0)
            if heat is not None:
                heat.set_data(np.zeros_like(heat.get_array()))
            return (
                [artist for artist in [heat, scatter, trail_scatter, lower_text, title_text, subtitle_text, credit_text, caption_text] if artist is not None]
            )

        idx = frame_info["index"]
        title_text.set_alpha(0.0)
        subtitle_text.set_alpha(0.0)
        credit_text.set_alpha(0.0)
        caption_text.set_alpha(caption_text.get_alpha())

        lon_now = dataset.lon.isel(time=idx).values if "lon" in dataset else np.array([])
        lat_now = dataset.lat.isel(time=idx).values if "lat" in dataset else np.array([])
        mask = np.isfinite(lon_now) & np.isfinite(lat_now)
        active_count = int(mask.sum())
        scatter.set_offsets(np.column_stack([lon_now[mask], lat_now[mask]]))
        current_colors = color_state["colors"]
        scatter.set_color(current_colors[mask])

        if settings.trails_length > 0 and n_particles:
            for pid in range(n_particles):
                if pid < lon_now.size and mask[pid]:
                    trail_history[pid].append((lon_now[pid], lat_now[pid]))
                else:
                    trail_history[pid].clear()
            trail_points: List[Tuple[float, float]] = []
            trail_colors: List[np.ndarray] = []
            for pid, history in enumerate(trail_history):
                if len(history) < 2:
                    continue
                base_rgba = np.array(color_state["rgba"][pid])
                for age, (lon_pt, lat_pt) in enumerate(reversed(history)):
                    trail_points.append((lon_pt, lat_pt))
                    trail_colors.append(_trail_rgba(base_rgba, age, len(history)))
            if trail_points:
                trail_scatter.set_offsets(np.array(trail_points))
                trail_scatter.set_facecolors(np.array(trail_colors))
                trail_scatter.set_alpha(1.0)
            else:
                trail_scatter.set_offsets(np.zeros((0, 2)))
                trail_scatter.set_alpha(0.0)

        if heat is not None and active_count:
            density = particle_density(
                lon_now[mask],
                lat_now[mask],
                config.domain,
                smooth_sigma=1.0,
            )
            heat.set_data(density.values)
            heat.set_extent(
                (
                    density.coords["lon"].values[0],
                    density.coords["lon"].values[-1],
                    density.coords["lat"].values[0],
                    density.coords["lat"].values[-1],
                )
            )

        lower_text.set_text(
            _lower_third_text(scenario_name, dataset, idx, active_count, config)
        )

        return (
            [artist for artist in [heat, scatter, trail_scatter, lower_text, title_text, subtitle_text, credit_text, caption_text] if artist is not None]
        )

    context: Dict[str, object] = {
        "ax": ax,
        "heat": heat,
        "scatter": scatter,
        "trail_scatter": trail_scatter,
        "lower_text": lower_text,
        "caption_text": caption_text,
        "title_text": title_text,
        "subtitle_text": subtitle_text,
        "credit_text": credit_text,
        "settings": settings,
        "config": config,
        "palette": palette,
        "color_state": color_state,
        "trail_history": trail_history,
        "dataset": dataset,
    }
    return AnimationScene(fig=fig, frames=frames, init_func=init, update_func=update, context=context)


def _create_dual_animation_scene(
    left: Tuple[SimulationConfig, xr.Dataset, str],
    right: Tuple[SimulationConfig, xr.Dataset, str],
    settings: AnimationSettings,
) -> AnimationScene:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=settings.figsize, subplot_kw={"projection": ccrs.PlateCarree()})
    panels = []
    for ax, (cfg, dataset, label) in zip(axes, [left, right]):
        ax.set_extent([cfg.domain.lon_min, cfg.domain.lon_max, cfg.domain.lat_min, cfg.domain.lat_max], ccrs.PlateCarree())
        ax.coastlines(resolution="50m", color="#d6d4c6", linewidth=0.4)
        ax.set_title(label)
        heat = None
        if settings.show_density:
            heat = ax.imshow(
                np.zeros((100, 100)),
                extent=(cfg.domain.lon_min, cfg.domain.lon_max, cfg.domain.lat_min, cfg.domain.lat_max),
                origin="lower",
                cmap="viridis",
                alpha=settings.density_alpha,
                zorder=1,
            )
        scatter = ax.scatter([], [], s=20, alpha=0.9, zorder=3, edgecolors="none")
        sources = dataset.coords.get("source_name")
        palette = _source_palette(sources.values if sources is not None else ["source"])
        if sources is not None:
            color_array = np.array([palette.get(src, "#ffffff") for src in sources.values])
        else:
            color_array = np.array(["#4e79a7"] * dataset.sizes.get("particle", 0))
        panels.append(
            {
                "ax": ax,
                "config": cfg,
                "dataset": dataset,
                "heat": heat,
                "scatter": scatter,
                "color_array": color_array,
            }
        )
    lower_text = fig.text(0.5, 0.03, "", ha="center", va="center", color="#f1f0ea", fontsize=12)
    frames = _prepare_frames(left[1], settings)

    def init_dual():
        artists = []
        for panel in panels:
            panel["scatter"].set_offsets(np.zeros((0, 2)))
            if panel["heat"] is not None:
                panel["heat"].set_data(np.zeros_like(panel["heat"].get_array()))
            artists.append(panel["scatter"])
            if panel["heat"] is not None:
                artists.append(panel["heat"])
        lower_text.set_text("")
        artists.append(lower_text)
        return [artist for artist in artists if artist is not None]

    def update_dual(frame_info: dict):
        kind = frame_info["kind"]
        if kind in {"title", "credits"}:
            lower_text.set_text("")
            return init_dual()
        idx = frame_info["index"]
        artists = []
        afloat_counts: List[int] = []
        gyre_counts: List[int] = []
        for panel in panels:
            dataset = panel["dataset"]
            cfg = panel["config"]
            max_idx = min(idx, dataset.sizes.get("time", 1) - 1)
            lon = dataset.lon.isel(time=max_idx).values if "lon" in dataset else np.array([])
            lat = dataset.lat.isel(time=max_idx).values if "lat" in dataset else np.array([])
            beached = dataset.beached.isel(time=max_idx).values if "beached" in dataset else np.array([])
            mask = np.isfinite(lon) & np.isfinite(lat)
            afloat_mask = mask & ~beached if beached.size else mask
            afloat_counts.append(int(np.count_nonzero(afloat_mask)))
            gyre_counts.append(int(np.count_nonzero(afloat_mask & _gyre_mask(lon, lat))))
            panel["scatter"].set_offsets(np.column_stack([lon[mask], lat[mask]]) if mask.any() else np.zeros((0, 2)))
            color_array = panel["color_array"]
            if color_array.size and mask.any():
                panel["scatter"].set_color(color_array[mask])
            if panel["heat"] is not None and mask.any():
                density = particle_density(lon[mask], lat[mask], cfg.domain, smooth_sigma=1.0)
                panel["heat"].set_data(density.values)
                panel["heat"].set_extent(
                    (
                        density.coords["lon"].values[0],
                        density.coords["lon"].values[-1],
                        density.coords["lat"].values[0],
                        density.coords["lat"].values[-1],
                    )
                )
            artists.append(panel["scatter"])
            if panel["heat"] is not None:
                artists.append(panel["heat"])
        time_values = left[1].time.values if "time" in left[1].coords else np.array([])
        if time_values.size:
            day = (time_values[min(idx, time_values.size - 1)] - time_values[0]) / np.timedelta64(1, "D")
            day_value = float(day)
        else:
            day_value = float(idx)
        lower_text.set_text(
            f"Day {day_value:.1f} | Ekman OFF: {afloat_counts[0]} afloat ({gyre_counts[0]} gyre) | Ekman ON: {afloat_counts[1]} afloat ({gyre_counts[1]} gyre)"
        )
        artists.append(lower_text)
        return [artist for artist in artists if artist is not None]

    return AnimationScene(
        fig=fig,
        frames=frames,
        init_func=init_dual,
        update_func=update_dual,
        context={"panels": panels, "lower_text": lower_text},
    )


def _render_frames(
    config: SimulationConfig,
    dataset: xr.Dataset,
    settings: AnimationSettings,
    output_path: Path,
    scenario_name: str,
) -> None:
    scene = create_animation_scene(config, dataset, settings, scenario_name)
    _write_scene(scene, settings, output_path, scenario_name)


def _write_scene(
    scene: AnimationScene,
    settings: AnimationSettings,
    output_path: Path,
    scenario_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(
        scene.fig,
        scene.update_func,
        frames=scene.frames,
        init_func=scene.init_func,
        blit=True,
    )

    writer_cfg: WriterConfig = safe_writer(settings.fps, settings.bitrate, settings.codec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if writer_cfg.backend == "matplotlib":
        writer = animation.FFMpegWriter(**writer_cfg.options)
        anim.save(output_path, writer=writer, dpi=settings.dpi)
        logger.info("Saved animation %s to %s via Matplotlib FFmpeg writer", scenario_name, output_path)
    else:
        import imageio

        with imageio.get_writer(output_path, **writer_cfg.options) as writer:
            for frame in scene.frames:
                scene.update_func(frame)
                scene.fig.canvas.draw()
                image = np.asarray(scene.fig.canvas.buffer_rgba())
                writer.append_data(image)
        logger.info("Saved animation %s to %s via imageio FFmpeg writer", scenario_name, output_path)
    plt.close(scene.fig)


def make_preview(
    config_path: Path | str,
    out: Path | str = "results/videos/preview.mp4",
    seed: Optional[int] = None,
) -> Path:
    """Generate the quick-preview animation."""
    cfg = load_config(config_path)
    dataset = run_simulation(cfg, seed=seed)
    out_path = Path(out)
    settings = preview_settings()
    _render_frames(cfg, dataset, settings, out_path, scenario_name=Path(config_path).stem)
    return out_path


def make_final_cut(
    config_path: Path | str,
    out: Path | str = "results/videos/final_cut.mp4",
    seed: Optional[int] = None,
) -> Path:
    """Generate the 2-3 minute final-cut animation."""
    cfg = load_config(config_path)
    dataset = run_simulation(cfg, seed=seed)
    out_path = Path(out)
    settings = final_settings()
    _render_frames(cfg, dataset, settings, out_path, scenario_name=Path(config_path).stem)
    return out_path


def animate_gyre_convergence(
    config_path: Path | str,
    days: int = 180,
    preset: str = "microplastic_default",
    out: Path | str = "results/videos/gyre_convergence.mp4",
    seed: Optional[int] = 42,
) -> Path:
    """Render focused convergence sequence highlighting subtropical accumulation."""
    cfg = _prepare_config(config_path, days, preset)
    dataset = run_simulation(cfg, seed=seed)
    dataset.attrs["preset_name"] = preset
    scenario = f"{Path(config_path).stem} – {preset}"
    settings = AnimationSettings(
        fps=24,
        frame_repeat=6,
        width_px=1920,
        height_px=1080,
        trails_length=18,
        title_seconds=2.5,
        credit_seconds=5.0,
        show_density=True,
        density_alpha=0.45,
        bitrate=8800,
    )
    out_path = Path(out)
    _render_frames(cfg, dataset, settings, out_path, scenario_name=scenario)
    return out_path


def animate_sources_mix(
    config_path: Path | str,
    days: int = 90,
    color_by_source: bool = True,
    legend_fade_in: bool = True,
    out: Path | str = "results/videos/sources_mix.mp4",
    seed: Optional[int] = 84,
) -> Path:
    """Render source-coloured animation with optional legend fade-in."""
    preset = "microplastic_default" if color_by_source else None
    cfg = _prepare_config(config_path, days, preset)
    dataset = run_simulation(cfg, seed=seed)
    dataset.attrs["preset_name"] = preset or "baseline"
    scenario = f"{Path(config_path).stem} – sources"
    settings = AnimationSettings(
        fps=24,
        frame_repeat=4,
        width_px=1280,
        height_px=720,
        trails_length=10,
        title_seconds=1.2,
        credit_seconds=3.5,
        show_density=True,
        density_alpha=0.25,
        bitrate=6500,
    )
    scene = create_animation_scene(cfg, dataset, settings, scenario_name=scenario)
    ax = scene.context["ax"]  # type: ignore[index]
    palette = scene.context.get("palette", {})
    if not color_by_source:
        color_state = scene.context.get("color_state")
        if isinstance(color_state, dict):
            base_color = "#76b7b2"
            color_state["colors"] = np.full(color_state["colors"].shape, base_color, dtype=object)  # type: ignore[index]
            color_state["rgba"] = np.array([mcolors.to_rgba(base_color) for _ in range(color_state["rgba"].shape[0])])  # type: ignore[index]
    handles: List[Line2D] = []
    labels: List[str] = []
    for name, color in palette.items():
        handles.append(Line2D([], [], marker="o", linestyle="None", markersize=6, color=color))
        labels.append(name.replace("_", " "))
    legend = ax.legend(handles, labels, loc="upper right", frameon=False) if handles else None
    if legend is not None:
        legend.set_alpha(0.0 if legend_fade_in else 0.95)

    original_update = scene.update_func
    total_frames = dataset.sizes.get("time", 1)
    fade_frames = max(int(total_frames * 0.1), 1)

    def update(frame_info: dict):
        artists = original_update(frame_info)
        if legend is not None:
            if frame_info["kind"] == "data":
                if legend_fade_in:
                    idx = frame_info["index"]
                    alpha = min(1.0, idx / fade_frames)
                    legend.set_alpha(alpha)
                artists.append(legend)
        return artists

    scene.update_func = update
    out_path = Path(out)
    _write_scene(scene, settings, out_path, scenario_name=scenario)
    return out_path


def animate_ekman_toggle(
    config_path: Path | str,
    days: int = 120,
    out: Path | str = "results/videos/ekman_toggle.mp4",
    seed: Optional[int] = 42,
) -> Path:
    """Render side-by-side animations with and without Ekman surface drift."""
    base_cfg = _prepare_config(config_path, days, "microplastic_default")
    ekman_cfg = _with_ekman(base_cfg, True)
    dataset_off = run_simulation(base_cfg, seed=seed)
    dataset_on = run_simulation(ekman_cfg, seed=seed)
    dataset_off.attrs["preset_name"] = dataset_off.attrs.get("preset_name", "ekman_off")
    dataset_on.attrs["preset_name"] = "ekman_on"
    settings = AnimationSettings(
        fps=24,
        frame_repeat=3,
        width_px=1920,
        height_px=1080,
        trails_length=0,
        title_seconds=2.0,
        credit_seconds=4.0,
        show_density=True,
        density_alpha=0.35,
        bitrate=7500,
    )
    scene = _create_dual_animation_scene(
        (base_cfg, dataset_off, "Ekman Disabled"),
        (ekman_cfg, dataset_on, "Ekman Enabled"),
        settings,
    )
    out_path = Path(out)
    _write_scene(scene, settings, out_path, scenario_name="ekman_toggle")
    return out_path


def animate_beaching_timelapse(
    config_path: Path | str,
    days: int = 90,
    out: Path | str = "results/videos/beaching_timelapse.mp4",
    seed: Optional[int] = 1337,
) -> Path:
    """Timelapse that persists beached particles along coastlines."""
    cfg = _prepare_config(config_path, days, "microplastic_default")
    dataset = run_simulation(cfg, seed=seed)
    dataset.attrs["preset_name"] = "beaching_timelapse"
    scenario = f"{Path(config_path).stem} – beaching"
    settings = AnimationSettings(
        fps=24,
        frame_repeat=4,
        width_px=1280,
        height_px=720,
        trails_length=6,
        title_seconds=1.5,
        credit_seconds=3.0,
        show_density=False,
        bitrate=6000,
    )
    scene = create_animation_scene(cfg, dataset, settings, scenario_name=scenario)
    ax = scene.context["ax"]  # type: ignore[index]
    dataset_ref: xr.Dataset = scene.context["dataset"]  # type: ignore[assignment]
    beach_scatter = ax.scatter([], [], s=32, marker="s", color="#e15759", edgecolors="#f1f0ea", linewidths=0.4, alpha=0.85, zorder=4)
    ticker = ax.text(
        0.02,
        0.9,
        "Beached: 0",
        transform=ax.transAxes,
        color="#f8d568",
        fontsize=13,
        bbox=dict(facecolor="#0b1d3a", alpha=0.65, pad=4, edgecolor="none"),
    )
    persistent: Dict[str, List[float]] = {"lon": [], "lat": []}
    seen: set[Tuple[int, int]] = set()
    original_update = scene.update_func

    def update(frame_info: dict):
        artists = original_update(frame_info)
        if frame_info["kind"] == "data":
            idx = frame_info["index"]
            lon_now = dataset_ref.lon.isel(time=idx).values
            lat_now = dataset_ref.lat.isel(time=idx).values
            beached = dataset_ref.beached.isel(time=idx).values
            mask = np.isfinite(lon_now) & np.isfinite(lat_now) & beached
            lon_vals = lon_now[mask]
            lat_vals = lat_now[mask]
            for lon_pt, lat_pt in zip(lon_vals, lat_vals):
                key = (int(round(lon_pt * 1000)), int(round(lat_pt * 1000)))
                if key not in seen:
                    persistent["lon"].append(float(lon_pt))
                    persistent["lat"].append(float(lat_pt))
                    seen.add(key)
            if persistent["lon"]:
                beach_scatter.set_offsets(np.column_stack([persistent["lon"], persistent["lat"]]))
            ticker.set_text(f"Beached: {len(persistent['lon'])}")
            artists.extend([beach_scatter, ticker])
        return artists

    scene.update_func = update
    out_path = Path(out)
    _write_scene(scene, settings, out_path, scenario_name=scenario)
    return out_path


def animate_parameter_sweep(
    sweep_configs: Iterable[Path | str],
    metric: str = "gyre_fraction",
    mosaic_cols: int = 3,
    out: Path | str = "results/videos/parameter_sweep.mp4",
    seed: Optional[int] = 21,
) -> Path:
    """Animate a grid of parameter sweep scenarios updating together."""
    configs = []
    for idx, cfg_path in enumerate(sweep_configs):
        if isinstance(cfg_path, SimulationConfig):
            cfg = cfg_path
        else:
            cfg = load_config(cfg_path)
        configs.append(cfg)
    if not configs:
        raise ValueError("No configurations provided for parameter sweep animation.")
    datasets: List[xr.Dataset] = []
    labels: List[str] = []
    for idx, cfg in enumerate(configs):
        run_seed = seed + idx if seed is not None else None
        ds = run_simulation(cfg, seed=run_seed)
        ds.attrs["preset_name"] = "sweep"
        datasets.append(ds)
        labels.append(f"wind={cfg.physics.windage_coeff:.3f}, Kh={cfg.physics.diffusivity_m2s:.0f}")
    max_time = max(ds.sizes.get("time", 0) for ds in datasets)
    rows = math.ceil(len(datasets) / mosaic_cols)
    cols = min(mosaic_cols, len(datasets))
    settings = AnimationSettings(
        fps=24,
        frame_repeat=2,
        width_px=1920,
        height_px=1080,
        trails_length=4,
        title_seconds=2.0,
        credit_seconds=4.0,
        show_density=False,
        bitrate=7000,
    )
    figsize = (settings.width_px / settings.dpi, settings.height_px / settings.dpi)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    apply_style()
    scatters: List[object] = []
    metric_texts: List[plt.Text] = []
    label_texts: List[plt.Text] = []
    domain = configs[0].domain
    idx_iter = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx_iter >= len(datasets):
                ax.set_visible(False)
                idx_iter += 1
                continue
            ax.set_extent([domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max], crs=ccrs.PlateCarree())
            ax.coastlines(resolution="50m", color="#d6d4c6", linewidth=0.45)
            scatter = ax.scatter([], [], s=12, color="#4e79a7", alpha=0.75, transform=ccrs.PlateCarree())
            label_text = ax.text(
                0.03,
                0.94,
                labels[idx_iter],
                transform=ax.transAxes,
                color="#f1f0ea",
                fontsize=9,
                bbox=dict(facecolor="#0b1d3a", alpha=0.6, pad=3, edgecolor="none"),
            )
            metric_text = ax.text(
                0.03,
                0.85,
                "",
                transform=ax.transAxes,
                color="#f8d568",
                fontsize=9,
            )
            scatters.append(scatter)
            label_texts.append(label_text)
            metric_texts.append(metric_text)
            idx_iter += 1
    title_text = fig.text(0.5, 0.93, "Parameter Sweep – Gyre Retention", ha="center", va="center", color="#f1f0ea", fontsize=20, weight="bold")
    subtitle_text = fig.text(0.5, 0.88, "Windage vs Diffusivity", ha="center", va="center", color="#f1f0ea", fontsize=14)
    credit_text = fig.text(0.5, 0.5, "Oceans Four Driftcast Sweep", ha="center", va="center", color="#f1f0ea", fontsize=18, alpha=0.0)

    frames: List[dict] = []
    frames.extend([{"kind": "title"}] * int(settings.title_seconds * settings.fps))
    for idx in range(max_time):
        frames.extend([{"kind": "data", "index": idx}] * settings.frame_repeat)
    frames.extend([{"kind": "credits"}] * int(settings.credit_seconds * settings.fps))

    def init():
        for scatter in scatters:
            scatter.set_offsets(np.zeros((0, 2)))
        title_text.set_alpha(1.0)
        subtitle_text.set_alpha(1.0)
        credit_text.set_alpha(0.0)
        return [title_text, subtitle_text, credit_text, *scatters, *label_texts, *metric_texts]

    def update(frame_info: dict):
        kind = frame_info["kind"]
        if kind == "title":
            title_text.set_alpha(1.0)
            subtitle_text.set_alpha(1.0)
            credit_text.set_alpha(0.0)
            return init()
        if kind == "credits":
            title_text.set_alpha(0.0)
            subtitle_text.set_alpha(0.0)
            credit_text.set_alpha(1.0)
            return [credit_text]
        idx = frame_info["index"]
        title_text.set_alpha(0.0)
        subtitle_text.set_alpha(0.0)
        credit_text.set_alpha(0.0)
        artists: List = []
        for ds, scatter, metric_text in zip(datasets, scatters, metric_texts):
            frame_idx = min(idx, ds.sizes.get("time", 1) - 1)
            lon_now = ds.lon.isel(time=frame_idx).values
            lat_now = ds.lat.isel(time=frame_idx).values
            beached = ds.beached.isel(time=frame_idx).values
            mask = np.isfinite(lon_now) & np.isfinite(lat_now) & ~beached
            if np.any(mask):
                scatter.set_offsets(np.column_stack([lon_now[mask], lat_now[mask]]))
            else:
                scatter.set_offsets(np.zeros((0, 2)))
            if metric == "gyre_fraction":
                inside = np.sum(_inside_gyre(lon_now[mask], lat_now[mask]))
                total = np.count_nonzero(mask)
                frac = inside / total if total else 0.0
                metric_text.set_text(f"Gyre frac: {frac:.2f}")
            artists.extend([scatter, metric_text])
        return artists

    scene = AnimationScene(fig=fig, frames=frames, init_func=init, update_func=update, context={"datasets": datasets})
    out_path = Path(out)
    _write_scene(scene, settings, out_path, scenario_name="parameter_sweep")
    return out_path


def animate_backtrack_from_gyre(
    config_path: Path | str,
    days_back: int = 30,
    out: Path | str = "results/videos/backtrack_from_gyre.mp4",
    seed: Optional[int] = 55,
) -> Path:
    """Highlight trajectories of particles ending inside the gyre by backtracking their history."""
    cfg = _prepare_config(config_path, None, "microplastic_default")
    dataset = run_simulation(cfg, seed=seed)
    dataset.attrs["preset_name"] = "backtrack"
    time_values = dataset.time.values
    if time_values.size > 1:
        dt_days = float((time_values[1] - time_values[0]) / np.timedelta64(1, "D"))
    else:
        dt_days = 1.0
    frames_back = min(dataset.sizes.get("time", 1), max(1, int(days_back / max(dt_days, 1e-6))))
    subset = dataset.isel(time=slice(-frames_back, None))
    final_lon = subset.lon.isel(time=-1).values
    final_lat = subset.lat.isel(time=-1).values
    mask = _inside_gyre(final_lon, final_lat) & np.isfinite(final_lon) & np.isfinite(final_lat)
    if not np.any(mask):
        mask = np.isfinite(final_lon) & np.isfinite(final_lat)
    subset = subset.isel(particle=np.where(mask)[0])
    scenario = f"{Path(config_path).stem} – backtrack"
    settings = AnimationSettings(
        fps=24,
        frame_repeat=4,
        width_px=1280,
        height_px=720,
        trails_length=max(frames_back // 3, 6),
        title_seconds=1.5,
        credit_seconds=3.0,
        show_density=False,
        bitrate=6200,
    )
    scene = create_animation_scene(cfg, subset, settings, scenario_name=scenario)
    color_state = scene.context.get("color_state")
    if isinstance(color_state, dict):
        highlight = "#00d4ff"
        color_state["colors"] = np.full(color_state["colors"].shape, highlight, dtype=object)  # type: ignore[index]
        color_state["rgba"] = np.array([mcolors.to_rgba(highlight) for _ in range(color_state["rgba"].shape[0])])  # type: ignore[index]
    out_path = Path(out)
    _write_scene(scene, settings, out_path, scenario_name=scenario)
    return out_path


def animate_long_cut(
    config_path: Path | str,
    preset: str = "microplastic_default",
    out: Path | str = "results/videos/natl_longcut.mp4",
    duration_minutes: float = 5.0,
    seed: Optional[int] = 42,
) -> Path:
    """Render the multi-scene long cut (2-10 minutes) with scripted overlays."""
    settings, scene, dataset, legend, scene_label, watermark, sections, frame_seconds, scenario = _prepare_longcut_scene(
        config_path, preset, duration_minutes, seed
    )
    _apply_longcut_overlays(
        scene,
        settings,
        dataset,
        legend,
        scene_label,
        watermark,
        sections,
        frame_seconds,
    )
    out_path = Path(out)
    _write_scene(scene, settings, out_path, scenario_name=scenario)
    return out_path


def animate_longcut_captions(
    config_path: Path | str,
    preset: str = "microplastic_default",
    out: Path | str = "results/videos/natl_longcut_captions.mp4",
    minutes: float = 5.0,
    captions: Optional[Path | str] = None,
    seed: Optional[int] = 42,
) -> Path:
    """Render the long cut with optional SRT captions overlaid."""
    settings, scene, dataset, legend, scene_label, watermark, sections, frame_seconds, scenario = _prepare_longcut_scene(
        config_path, preset, minutes, seed
    )
    caption_segments: Optional[List[Tuple[float, float, str]]] = None
    caption_text_obj = scene.context.get("caption_text")
    caption_text = caption_text_obj if isinstance(caption_text_obj, plt.Text) else None  # type: ignore[assignment]
    if captions is not None:
        caption_path = Path(captions)
        if caption_path.exists():
            caption_segments = _parse_srt(caption_path)
    _apply_longcut_overlays(
        scene,
        settings,
        dataset,
        legend,
        scene_label,
        watermark,
        sections,
        frame_seconds,
        caption_text=caption_text,
        captions=caption_segments,
    )
    out_path = Path(out)
    _write_scene(scene, settings, out_path, scenario_name=f"{scenario} (captions)")
    return out_path
