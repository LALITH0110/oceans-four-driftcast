# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Detects FFmpeg binaries and provides safe writer configurations for animations.
- Chooses between Matplotlib's FFMpegWriter and imageio-ffmpeg fallbacks.
- Surfaces actionable installation guidance when FFmpeg is missing.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Literal

import matplotlib as mpl
from matplotlib import animation


@dataclass(frozen=True)
class WriterConfig:
    """Structured configuration describing the selected video writer backend."""

    backend: Literal["matplotlib", "imageio"]
    options: Dict[str, Any]


def detect_ffmpeg() -> str:
    """Return the FFmpeg executable path or raise with platform-specific guidance."""
    candidate = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if candidate:
        mpl.rcParams["animation.ffmpeg_path"] = candidate
        return candidate

    system = platform.system().lower()
    if system == "windows":
        hint = "Install FFmpeg via scoop (`scoop install ffmpeg`) or add ffmpeg.exe to PATH."
    elif system == "darwin":
        hint = "Install FFmpeg via Homebrew (`brew install ffmpeg`)."
    else:
        hint = "Install FFmpeg via your package manager (e.g., `sudo apt install ffmpeg`)."
    raise FileNotFoundError(f"FFmpeg binary not found on PATH. {hint}")


def safe_writer(fps: int, bitrate: int, codec: str) -> WriterConfig:
    """Return a writer configuration prioritizing Matplotlib and falling back to imageio."""
    detect_ffmpeg()
    if animation.writers.is_available("ffmpeg"):
        return WriterConfig(
            backend="matplotlib",
            options={"fps": fps, "bitrate": bitrate, "codec": codec, "extra_args": ["-loglevel", "error"]},
        )

    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in CI failures
        raise RuntimeError(
            "Matplotlib FFmpeg writer unavailable and imageio-ffmpeg not installed. "
            "Install `imageio-ffmpeg` to enable the fallback writer."
        ) from exc

    return WriterConfig(
        backend="imageio",
        options={"fps": fps, "codec": codec, "bitrate": bitrate, "ffmpeg_log_level": "error"},
    )
