"""
File Summary:
- Packs key Driftcast artifacts into a judge-friendly release directory.
- Copies hero imagery, highlight videos, top figures, validation report, and docs.
- Provides a single command invoked via the CLI publish group.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple


DEFAULT_VIDEO_CANDIDATES: List[Path] = [
    Path("results/videos/natl_longcut.mp4"),
    Path("results/videos/final_cut.mp4"),
    Path("results/videos/preview.mp4"),
]

DEFAULT_OPTIONAL_ASSETS: List[Tuple[Path, Path]] = [
    (Path("results/videos/preview.gif"), Path("videos/preview.gif")),
    (Path("results/validation/report.json"), Path("validation/report.json")),
]


def _copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _select_top_figures(figures_dir: Path, limit: int = 12) -> List[Path]:
    if not figures_dir.exists():
        return []
    pngs = sorted(figures_dir.glob("*.png"))
    return pngs[:limit]


def make_release_bundle(out_dir: Path | str = Path("release")) -> Path:
    """Create a curated release bundle for judges and stakeholders."""
    out_path = Path(out_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(Path("results/figures/hero.png"), out_path / "hero.png")
    _copy_if_exists(Path("docs/onepager.pdf"), out_path / "docs" / "onepager.pdf")

    for candidate in DEFAULT_VIDEO_CANDIDATES:
        if _copy_if_exists(candidate, out_path / "videos" / candidate.name):
            break

    for src, dest in DEFAULT_OPTIONAL_ASSETS:
        _copy_if_exists(src, out_path / dest)

    figures = _select_top_figures(Path("results/figures"))
    for fig in figures:
        _copy_if_exists(fig, out_path / "figures" / fig.name)

    readme_src = Path("README_release.md")
    if readme_src.exists():
        _copy_if_exists(readme_src, out_path / "README_release.md")

    return out_path
