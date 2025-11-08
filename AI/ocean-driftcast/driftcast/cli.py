# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Typer-based command line interface for running simulations and utilities.
- Offers commands for single runs, batch sweeps, animations, ingest QA, perf checks, and judge packaging.
- Entry point exposed as ``driftcast`` via the pyproject console script.
"""

from __future__ import annotations

import random
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import typer
from matplotlib import pyplot as plt
import io

from driftcast import configure_logging, logger
from driftcast.config import SimulationConfig, load_config
from driftcast.ingest.normalize import ingest_json_file, validate_json_file
from driftcast.ingest.schema import CrowdSchema
from driftcast.sim.batch import BatchRunner
from driftcast.sim.runner import run_simulation
from driftcast.viz.animate import (
    create_animation_scene,
    final_settings,
    make_final_cut,
    make_preview,
    animate_gyre_convergence,
    animate_sources_mix,
    animate_beaching_timelapse,
    animate_parameter_sweep,
    animate_backtrack_from_gyre,
    animate_long_cut,
    animate_ekman_toggle,
    animate_longcut_captions,
    _render_frames,
)
from driftcast.viz import plots as viz_plots
from driftcast.validate import compute_golden_numbers, assert_sane, write_validation_report
from driftcast.tools.bundle import make_release_bundle

app = typer.Typer(help="driftcast simulation toolkit")
animate_app = typer.Typer(help="Render animations from simulation outputs")
plots_app = typer.Typer(help="Generate static figures for Driftcast runs")
validate_app = typer.Typer(help="Validation and golden-number utilities")
ingest_app = typer.Typer(help="Crowdsourced ingest utilities")
perf_app = typer.Typer(help="Performance diagnostics")
publish_app = typer.Typer(help="Release packaging helpers")

app.add_typer(animate_app, name="animate")
app.add_typer(plots_app, name="plots")
app.add_typer(validate_app, name="validate")
app.add_typer(ingest_app, name="ingest")
app.add_typer(perf_app, name="perf")
app.add_typer(publish_app, name="publish")


def _apply_overrides(cfg: SimulationConfig, overrides: Dict[str, float]) -> SimulationConfig:
    payload = cfg.model_dump()
    for dotted_key, value in overrides.items():
        node = payload
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return SimulationConfig(**payload)


def _parse_param_flags(param: Iterable[str]) -> List[Dict[str, float]]:
    if not param:
        return []
    grid: Dict[str, List[float]] = {}
    for spec in param:
        if "=" not in spec:
            raise typer.BadParameter(f"Invalid --param spec '{spec}'")
        key, values = spec.split("=", 1)
        parsed = [float(val) for val in values.split(",")]
        grid[key] = parsed
    keys = list(grid.keys())
    combos: List[Dict[str, float]] = []
    import itertools

    for product in itertools.product(*(grid[key] for key in keys)):
        combos.append(dict(zip(keys, product)))
    return combos


def _normalize_sweep_params(param: Iterable[str]) -> List[str]:
    """Map short-form parameter aliases to dotted SimulationConfig paths."""
    mapping = {
        "windage": "physics.windage_coeff",
        "kh": "physics.diffusivity_m2s",
        "diffusivity": "physics.diffusivity_m2s",
        "stokes": "physics.stokes_coeff",
    }
    normalized: List[str] = []
    for spec in param:
        if "=" not in spec:
            raise typer.BadParameter(f"Invalid --param spec '{spec}'")
        key, values = spec.split("=", 1)
        mapped = mapping.get(key.lower(), key)
        normalized.append(f"{mapped}={values}")
    return normalized


@plots_app.command("all")
def plots_all(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=True, dir_okay=True, help="Path to NetCDF/Zarr output or directory."),
    config: Path = typer.Option(..., "--config", exists=True, help="Configuration YAML used for the run (for schedule/streamfunction)."),
    sweep: Optional[Path] = typer.Option(None, "--sweep", exists=True, help="Optional sweep directory for parameter matrix."),
    stream_day: Optional[float] = typer.Option(None, "--stream-day", help="Day (since start) to sample the streamfunction contours."),
    compare_run: List[Path] = typer.Option([], "--compare-run", exists=True, file_okay=True, dir_okay=True, help="Additional run paths for preset comparison."),
    metric: str = typer.Option("gyre_fraction", "--metric", help="Metric for sweep matrix colour scaling."),
) -> None:
    """Generate the full publication-grade figure set."""
    cfg = load_config(config)
    produced: List[str] = []

    def _close(fig: plt.Figure) -> None:
        plt.close(fig)

    fig = viz_plots.plot_accumulation_heatmap(run)
    _close(fig)
    produced.append("accumulation_heatmap")

    fig = viz_plots.plot_source_mix_pie(run)
    _close(fig)
    produced.append("source_mix_pie")

    fig = viz_plots.plot_source_contribution_map(run)
    _close(fig)
    produced.append("source_contribution_map")

    fig = viz_plots.plot_beaching_hotspots(run)
    _close(fig)
    produced.append("beaching_hotspots")

    fig = viz_plots.plot_residence_time(run)
    _close(fig)
    produced.append("residence_time")

    fig = viz_plots.plot_age_histogram(run)
    _close(fig)
    produced.append("age_histogram")

    fig = viz_plots.plot_time_series(run)
    _close(fig)
    produced.append("time_series")

    fig = viz_plots.plot_hovmoller_lat_density(run)
    _close(fig)
    produced.append("hovmoller_lat_density")

    fig = viz_plots.plot_traj_bundle(run)
    _close(fig)
    produced.append("traj_bundle")

    fig = viz_plots.plot_curvature_map(run)
    _close(fig)
    produced.append("curvature_map")

    fig = viz_plots.plot_density_vs_distance_to_gyre_center(run)
    _close(fig)
    produced.append("density_vs_distance")

    fig = viz_plots.plot_hotspot_rank(run)
    _close(fig)
    produced.append("hotspot_rank")

    sample_day = stream_day if stream_day is not None else cfg.time.duration_days / 2.0
    fig = viz_plots.plot_streamfunction_contours(config, t=sample_day)
    _close(fig)
    produced.append("streamfunction_contours")

    fig = viz_plots.plot_release_schedule(config)
    _close(fig)
    produced.append("release_schedule")

    if sweep:
        fig = viz_plots.plot_parameter_sweep_matrix(sweep, metric=metric)
        _close(fig)
        produced.append("parameter_sweep_matrix")

    if compare_run:
        compare_inputs = [run, *compare_run]
        fig = viz_plots.plot_compare_presets(compare_inputs)
        _close(fig)
        produced.append("compare_presets")

    typer.echo(f"Generated {len(produced)} figures in results/figures and docs/assets.")


@plots_app.command("key")
def plots_key(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=True, dir_okay=True, help="Path to NetCDF/Zarr output or directory."),
    config: Path = typer.Option(..., "--config", exists=True, help="Configuration YAML used for the run."),
) -> None:
    """Generate the short list of hero figures for judging."""
    cfg = load_config(config)

    def _close(fig: plt.Figure) -> None:
        plt.close(fig)

    sample_day = cfg.time.duration_days / 2.0
    _close(viz_plots.plot_accumulation_heatmap(run))
    _close(viz_plots.plot_time_series(run))
    _close(viz_plots.plot_source_mix_pie(run))
    _close(viz_plots.plot_beaching_hotspots(run))
    _close(viz_plots.plot_streamfunction_contours(config, t=sample_day))
    _close(viz_plots.plot_density_vs_distance_to_gyre_center(run))
    typer.echo("Generated key figure set.")


@plots_app.command("extra")
def plots_extra(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=True, dir_okay=True, help="Reference run for extra diagnostics."),
    ekman_run: List[Path] = typer.Option([], "--ekman-run", exists=True, file_okay=True, dir_okay=True, help="Two run paths (off,on) for Ekman comparison."),
    seasonal_run: List[Path] = typer.Option([], "--seasonal-run", exists=True, file_okay=True, dir_okay=True, help="Two run paths (off,on) for seasonal ramp comparison."),
) -> None:
    """Generate supplementary figures for validation and storytelling."""

    def _close(fig: plt.Figure) -> None:
        plt.close(fig)

    _close(viz_plots.plot_gyre_fraction_curve(run))
    _close(viz_plots.plot_curvature_cdf(run))
    if ekman_run:
        if len(ekman_run) != 2:
            raise typer.BadParameter("--ekman-run expects exactly two paths (off,on).")
        _close(viz_plots.plot_ekman_vs_noekman(ekman_run))
    if seasonal_run:
        if len(seasonal_run) != 2:
            raise typer.BadParameter("--seasonal-run expects exactly two paths (off,on).")
        _close(viz_plots.plot_seasonal_ramp_effect(seasonal_run))
    typer.echo("Generated extra diagnostics (gyre fraction, curvature CDF, comparisons where provided).")


@validate_app.command("run")
def validate_run(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=True, dir_okay=True, help="Path to simulation output (file or directory)."),
    out: Path = typer.Option(Path("results/validation/report.json"), "--out", help="Validation report destination."),
) -> None:
    """Compute golden numbers, emit a report, and assert sanity thresholds."""
    report_path = write_validation_report(run, out)
    metrics = compute_golden_numbers(run)
    try:
        assert_sane(run)
    except AssertionError as exc:
        typer.echo(f"Validation report written to {report_path.resolve()} but thresholds failed: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"Validation report written to {report_path.resolve()}")
    typer.echo("Golden numbers:")
    for key, value in metrics.items():
        typer.echo(f"  {key}: {value:.4f}")


@publish_app.command("bundle")
def publish_bundle(
    out: Path = typer.Option(Path("release"), "--out", help="Destination directory for the release bundle."),
) -> None:
    """Create a curated release bundle with hero media, figures, docs, and validation outputs."""
    dest = make_release_bundle(out)
    typer.echo(f"Release bundle written to {dest.resolve()}")

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging."),
) -> None:
    """Configure logging before executing commands."""
    level = "DEBUG" if verbose else "INFO"
    configure_logging(level=level)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)


@app.command("run")
def run(
    config: Path = typer.Argument(..., exists=True, help="Path to YAML config."),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Deterministic seed for numpy and random modules."
    ),
) -> None:
    """Run a single simulation defined by a YAML configuration file."""
    cfg = load_config(config)
    _seed_everything(seed)
    dataset = run_simulation(cfg, seed=seed)
    logger.info("Run completed with %d particles", dataset.sizes.get("particle", 0))


@app.command("sweep")
def sweep(
    config: Path = typer.Argument(..., exists=True, help="Base YAML config."),
    param: List[str] = typer.Option([], "--param", help="Override e.g. physics.diffusivity_m2s=5,10"),
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Dask scheduler address."),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Deterministic base seed for the parameter sweep."
    ),
) -> None:
    """Run a batch sweep over parameter combinations."""
    base_cfg = load_config(config)
    overrides = _parse_param_flags(param)
    configs = [base_cfg] if not overrides else []
    if overrides:
        for override in overrides:
            configs.append(_apply_overrides(base_cfg, override))
    _seed_everything(seed)
    runner = BatchRunner(configs=configs, base_seed=seed)
    if cluster:
        runner.use_distributed = True
        runner.cluster_address = cluster
    runner.run()


@animate_app.command("preview")
def animate_preview(
    config: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("results/videos/preview.mp4"), "--out"),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Optional deterministic seed forwarded to the simulation."
    ),
) -> None:
    """Render the short preview animation."""
    _seed_everything(seed)
    make_preview(config, out=out, seed=seed)


@animate_app.command("final")
def animate_final(
    config: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("results/videos/final_cut.mp4"), "--out"),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Optional deterministic seed forwarded to the simulation."
    ),
) -> None:
    """Render the full-length competition animation."""
    _seed_everything(seed)
    make_final_cut(config, out=out, seed=seed)


@animate_app.command("gyre")
def animate_gyre(
    config: Path = typer.Option(Path("configs/natl_subtropical_gyre.yaml"), "--config", exists=True, help="Configuration YAML to simulate."),
    days: int = typer.Option(180, "--days", help="Simulation horizon in days."),
    preset: str = typer.Option("microplastic_default", "--preset", help="Physics preset to apply."),
    out: Path = typer.Option(Path("results/videos/gyre_convergence.mp4"), "--out", help="Output MP4 path."),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Render the gyre convergence spotlight animation."""
    _seed_everything(seed)
    animate_gyre_convergence(config, days=days, preset=preset, out=out, seed=seed)
    typer.echo(f"Saved gyre convergence animation to {out.resolve()}")


@animate_app.command("sources")
def animate_sources(
    config: Path = typer.Option(Path("configs/natl_subtropical_gyre.yaml"), "--config", exists=True, help="Configuration YAML to simulate."),
    days: int = typer.Option(90, "--days", help="Simulation horizon in days."),
    color_by_source: bool = typer.Option(True, "--color-by-source/--no-color-by-source", help="Colour particles by source class."),
    legend_fade_in: bool = typer.Option(True, "--legend-fade-in/--legend-static", help="Fade legend in over time."),
    out: Path = typer.Option(Path("results/videos/sources_mix.mp4"), "--out", help="Output MP4 path."),
    seed: Optional[int] = typer.Option(84, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Render the source mix animation with optional legend fade."""
    _seed_everything(seed)
    animate_sources_mix(
        config,
        days=days,
        color_by_source=color_by_source,
        legend_fade_in=legend_fade_in,
        out=out,
        seed=seed,
    )
    typer.echo(f"Saved sources mix animation to {out.resolve()}")


@animate_app.command("beaching")
def animate_beaching(
    config: Path = typer.Option(Path("configs/natl_subtropical_gyre.yaml"), "--config", exists=True, help="Configuration YAML to simulate."),
    days: int = typer.Option(90, "--days", help="Simulation horizon in days."),
    out: Path = typer.Option(Path("results/videos/beaching_timelapse.mp4"), "--out", help="Output MP4 path."),
    seed: Optional[int] = typer.Option(1337, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Render the beaching timelapse animation with persistent hotspots."""
    _seed_everything(seed)
    animate_beaching_timelapse(config, days=days, out=out, seed=seed)
    typer.echo(f"Saved beaching timelapse to {out.resolve()}")


@animate_app.command("backtrack")
def animate_backtrack(
    config: Path = typer.Option(Path("configs/natl_subtropical_gyre.yaml"), "--config", exists=True, help="Configuration YAML to simulate."),
    days_back: int = typer.Option(30, "--days-back", help="Number of days to trace backwards from gyre core."),
    out: Path = typer.Option(Path("results/videos/backtrack_from_gyre.mp4"), "--out", help="Output MP4 path."),
    seed: Optional[int] = typer.Option(55, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Render the backtracking animation from the gyre core."""
    _seed_everything(seed)
    animate_backtrack_from_gyre(config, days_back=days_back, out=out, seed=seed)
    typer.echo(f"Saved backtrack animation to {out.resolve()}")


@animate_app.command("ekman")
def animate_ekman(
    config: Path = typer.Option(Path("configs/natl_subtropical_gyre.yaml"), "--config", exists=True, help="Configuration YAML to simulate."),
    days: int = typer.Option(120, "--days", help="Simulation horizon in days."),
    out: Path = typer.Option(Path("results/videos/ekman_toggle.mp4"), "--out", help="Output MP4 path."),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Render Ekman-off versus Ekman-on comparison panels."""
    _seed_everything(seed)
    animate_ekman_toggle(config, days=days, out=out, seed=seed)
    typer.echo(f"Saved Ekman toggle animation to {out.resolve()}")


@animate_app.command("long")
def animate_long(
    config: Path = typer.Option(Path("configs/natl_subtropical_gyre.yaml"), "--config", exists=True, help="Configuration YAML to simulate."),
    minutes: float = typer.Option(5.0, "--minutes", help="Target duration in minutes (2-10)."),
    preset: str = typer.Option("microplastic_default", "--preset", help="Physics preset to apply."),
    out: Path = typer.Option(Path("results/videos/natl_longcut.mp4"), "--out", help="Output MP4 path."),
    captions: Optional[Path] = typer.Option(None, "--captions", exists=True, help="Optional SRT caption file."),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Render the scripted long cut animation."""
    _seed_everything(seed)
    if captions is not None:
        animate_longcut_captions(config, preset=preset, out=out, minutes=minutes, captions=captions, seed=seed)
    else:
        animate_long_cut(config, preset=preset, out=out, duration_minutes=minutes, seed=seed)
    typer.echo(f"Saved long cut animation to {out.resolve()}")


@animate_app.command("sweep")
def animate_sweep_video(
    config: Path = typer.Option(Path("configs/natl_subtropical_gyre.yaml"), "--config", exists=True, help="Base configuration YAML."),
    param: List[str] = typer.Option([], "--param", help="Parameter overrides (e.g. windage=0.001,0.005)."),
    metric: str = typer.Option("gyre_fraction", "--metric", help="Metric for colouring the matrix."),
    mosaic_cols: int = typer.Option(3, "--cols", help="Number of columns in the mosaic."),
    out: Path = typer.Option(Path("results/videos/parameter_sweep.mp4"), "--out", help="Output MP4 path."),
    seed: Optional[int] = typer.Option(21, "--seed", help="Seed forwarded to the simulations."),
) -> None:
    """Render a parameter sweep mosaic animation."""
    base_cfg = load_config(config)
    overrides = _parse_param_flags(_normalize_sweep_params(param))
    configs = [base_cfg] if not overrides else [_apply_overrides(base_cfg, override) for override in overrides]
    _seed_everything(seed)
    animate_parameter_sweep(configs, metric=metric, mosaic_cols=mosaic_cols, out=out, seed=seed)
    typer.echo(f"Saved parameter sweep animation to {out.resolve()}")


@ingest_app.command("normalize")
def ingest_normalize(
    json_path: Path = typer.Argument(..., exists=True, help="Crowdsourced JSON observations."),
    schema_path: Path = typer.Option(
        Path("schemas/crowd_drifters.schema.json"), "--schema", help="JSON schema path."
    ),
    out_dir: Path = typer.Option(
        Path("data/crowd/processed"), "--out-dir", help="Parquet output directory."
    ),
) -> None:
    """Validate and normalize observations into partitioned parquet."""
    schema = CrowdSchema.load(schema_path)
    output = ingest_json_file(json_path, schema=schema, output_dir=out_dir)
    typer.echo(f"Normalized ingest written to {output.resolve()}")


# Provide backwards-compatible alias for normalize.
ingest_app.command("run")(ingest_normalize)  # type: ignore[arg-type]


@app.command("ingest")
def ingest_default(
    json_path: Path = typer.Argument(..., exists=True, help="Crowdsourced JSON observations."),
    schema_path: Path = typer.Option(
        Path("schemas/crowd_drifters.schema.json"), "--schema", help="JSON schema path."
    ),
    out_dir: Path = typer.Option(
        Path("data/crowd/processed"), "--out-dir", help="Parquet output directory."
    ),
) -> None:
    """Backwards-compatible alias that normalizes observations."""
    ingest_normalize(json_path=json_path, schema_path=schema_path, out_dir=out_dir)

@ingest_app.command("validate")
def ingest_validate(
    json_path: Path = typer.Option(..., "--json", exists=True, help="Crowdsourced JSON file."),
    schema_path: Path = typer.Option(
        Path("schemas/crowd_drifters.schema.json"), "--schema", help="JSON schema path."
    ),
) -> None:
    """Validate a JSON payload without writing outputs."""
    schema = CrowdSchema.load(schema_path)
    records, duplicates_removed = validate_json_file(json_path, schema=schema)
    typer.echo(
        f"Validation successful: {records} records after removing {duplicates_removed} duplicates."
    )


def _short_config_for_perf(cfg: SimulationConfig) -> SimulationConfig:
    payload = cfg.model_dump()
    payload["time"]["end"] = (cfg.time.start + timedelta(seconds=5)).isoformat()
    payload["time"]["dt_minutes"] = max(payload["time"]["dt_minutes"], 5.0 / 60.0)
    return SimulationConfig(**payload)


@perf_app.command("check")
def perf_check(
    config: Path = typer.Option(
        Path("configs/natl_subtropical_gyre.yaml"), "--config", help="Configuration to benchmark."
    ),
    seed: Optional[int] = typer.Option(123, "--seed", help="Deterministic seed for the benchmark run."),
) -> None:
    """Run a quick 5-second simulation to report FPS and peak memory."""
    cfg = load_config(config)
    short_cfg = _short_config_for_perf(cfg)
    _seed_everything(seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "perf.nc"
        tracemalloc.start()
        start = time.perf_counter()
        ds = run_simulation(short_cfg, output_path=tmp_path, seed=seed, write_manifest_sidecar=False)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    frames = max(ds.sizes.get("time", 1), 1)
    fps = frames / elapsed if elapsed > 0 else float("inf")
    typer.echo(
        f"Perf check -> frames: {frames}, elapsed: {elapsed:.2f}s, FPS: {fps:.2f}, "
        f"peak memory: {peak / 1e6:.1f} MB"
    )


@app.command("judge")
def judge(
    config: Path = typer.Option(
        Path("configs/natl_subtropical_gyre.yaml"), "--config", help="Scenario used for the judge package."
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed for judge deliverables."),
) -> None:
    """Generate the judge-ready package: data, final MP4, hero PNG, and one-pager PDF."""
    typer.echo("Preparing synthetic inputs...")
    subprocess.run([sys.executable, "scripts/generate_synthetic_inputs.py"], check=True)
    schema = CrowdSchema.load("schemas/crowd_drifters.schema.json")
    ingest_json_file(Path("data/raw/mock_crowd.json"), schema=schema, output_dir=Path("data/crowd/processed"))

    typer.echo("Running simulation...")
    cfg = load_config(config)
    _seed_everything(seed)
    dataset = run_simulation(cfg, seed=seed)

    scenario_name = Path(config).stem
    final_video = Path("results/videos/final_cut.mp4")
    final_video.parent.mkdir(parents=True, exist_ok=True)
    settings = final_settings()
    _render_frames(cfg, dataset, settings, final_video, scenario_name=scenario_name)

    typer.echo("Rendering hero frame...")
    hero_path = Path("results/figures/hero.png")
    hero_path.parent.mkdir(parents=True, exist_ok=True)
    scene = create_animation_scene(cfg, dataset, settings, scenario_name=scenario_name)
    scene.init_func()

    def _grab_frame_png() -> "np.ndarray":
        """Capture the current figure as an RGB array via an in-memory PNG."""
        buf = io.BytesIO()
        scene.fig.savefig(buf, format="png", dpi=scene.fig.dpi)
        buf.seek(0)
        image = plt.imread(buf)
        buf.close()
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        return image

    target_frame = None
    hero_index = max(dataset.sizes.get("time", 1) // 2, 0)
    for frame in scene.frames:
        scene.update_func(frame)
        if frame["kind"] == "data" and frame["index"] >= hero_index:
            target_frame = _grab_frame_png()
            break
    if target_frame is None:
        target_frame = _grab_frame_png()
    plt.imsave(hero_path, target_frame)
    plt.close(scene.fig)

    typer.echo("Building one-pager PDF...")
    subprocess.run([sys.executable, "scripts/render_onepager_pdf.py"], check=True)
    pdf_path = Path("docs/onepager.pdf")

    typer.echo(
        "\nJudge deliverables:\n"
        f"  Video: {final_video.resolve()}\n"
        f"  Hero:  {hero_path.resolve()}\n"
        f"  PDF:   {pdf_path.resolve()}\n"
    )


def entrypoint() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    entrypoint()
