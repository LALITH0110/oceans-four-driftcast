# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Provides utilities for launching batches of simulations via Dask.
- Wraps run_simulation in lazy delayed tasks and persists outputs concurrently.
- Exposed through the CLI sweep command for parameter sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import dask
from driftcast import logger

from driftcast.config import SimulationConfig
from .runner import run_simulation


@dataclass
class BatchRunner:
    """Batch orchestration for multiple simulation configurations."""

    configs: Iterable[SimulationConfig]
    output_dir: Path = Path("results/batch")
    use_distributed: bool = False
    cluster_address: Optional[str] = None
    base_seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.configs = list(self.configs)

    def build_tasks(self) -> List[Tuple[dask.delayed, Path]]:
        """Return a list of delayed simulation tasks with output paths."""
        tasks: List[tuple[dask.delayed, Path]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for idx, cfg in enumerate(self.configs):
            output_path = self.output_dir / f"{cfg.output.directory.name}_{cfg.time.start:%Y%m%d}.nc"
            seed = self.base_seed + idx if self.base_seed is not None else None
            task = dask.delayed(run_simulation)(cfg, output_path=output_path, seed=seed)
            tasks.append((task, output_path))
        return tasks

    def run(self) -> List[Path]:
        """Execute all configured simulations with Dask."""
        task_pairs = self.build_tasks()
        logger.info("Launching %d simulations via Dask", len(task_pairs))
        if not task_pairs:
            return []
        tasks, paths = zip(*task_pairs)
        dask.compute(*tasks)
        return list(paths)
