# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Centralizes matplotlib style customizations for driftcast figures.
- Provides apply_style() utility used across map and animation modules.
- Ensures cohesive typography, colors, and background aesthetics.
"""

from __future__ import annotations

from matplotlib import pyplot as plt


def apply_style() -> None:
    """Register the driftcast matplotlib style globally."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1d3a",
            "axes.facecolor": "#0b1d3a",
            "axes.edgecolor": "#d6d4c6",
            "axes.labelcolor": "#f1f0ea",
            "xtick.color": "#f1f0ea",
            "ytick.color": "#f1f0ea",
            "font.size": 12,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "#0b1d3a",
            "savefig.dpi": 150,
        }
    )
