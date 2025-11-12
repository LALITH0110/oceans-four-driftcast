"""
File Summary:
- Hosts visualization utilities for styling, static maps, and animations.
- Re-exports primary user-facing functions for quick notebook usage.
- See animate.make_preview/make_final_cut for MP4 rendering pipelines.
"""

from .style import apply_style
from .map import make_basemap
from .animate import (
    make_final_cut,
    make_preview,
    animate_gyre_convergence,
    animate_sources_mix,
    animate_beaching_timelapse,
    animate_ekman_toggle,
    animate_parameter_sweep,
    animate_backtrack_from_gyre,
    animate_long_cut,
    animate_longcut_captions,
)
from .plots import (
    plot_accumulation_heatmap,
    plot_streamfunction_contours,
    plot_source_mix_pie,
    plot_source_contribution_map,
    plot_beaching_hotspots,
    plot_residence_time,
    plot_age_histogram,
    plot_time_series,
    plot_hovmoller_lat_density,
    plot_parameter_sweep_matrix,
    plot_traj_bundle,
    plot_curvature_map,
    plot_release_schedule,
    plot_density_vs_distance_to_gyre_center,
    plot_hotspot_rank,
    plot_compare_presets,
    plot_gyre_fraction_curve,
    plot_curvature_cdf,
    plot_ekman_vs_noekman,
    plot_seasonal_ramp_effect,
)

__all__ = [
    "apply_style",
    "make_basemap",
    "make_preview",
    "make_final_cut",
    "animate_gyre_convergence",
    "animate_sources_mix",
    "animate_beaching_timelapse",
    "animate_ekman_toggle",
    "animate_parameter_sweep",
    "animate_backtrack_from_gyre",
    "animate_long_cut",
    "animate_longcut_captions",
    "plot_accumulation_heatmap",
    "plot_streamfunction_contours",
    "plot_source_mix_pie",
    "plot_source_contribution_map",
    "plot_beaching_hotspots",
    "plot_residence_time",
    "plot_age_histogram",
    "plot_time_series",
    "plot_hovmoller_lat_density",
    "plot_parameter_sweep_matrix",
    "plot_traj_bundle",
    "plot_curvature_map",
    "plot_release_schedule",
    "plot_density_vs_distance_to_gyre_center",
    "plot_hotspot_rank",
    "plot_compare_presets",
    "plot_gyre_fraction_curve",
    "plot_curvature_cdf",
    "plot_ekman_vs_noekman",
    "plot_seasonal_ramp_effect",
]
