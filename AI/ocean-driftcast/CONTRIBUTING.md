File Summary:
- Contribution guidelines covering environment setup, style, testing, and docs.
- Explains branching strategy, commit messaging, and pre-commit enforcement.
- Points new contributors toward key modules and onboarding resources.

# Contributing to Driftcast

## Local Setup

1. Create and activate the Conda env:
   ```bash
   make env
   conda activate driftcast
   pip install -e .[dev]
   ```
2. Install pre-commit hooks:
   ```bash
   make precommit
   ```
3. Run the quickstart simulation to verify your environment:
   ```bash
   make run
   ```

## Code Style

- Python formatting is enforced by `black`, `isort`, `ruff`, and `docformatter`. Hooks run automatically on commit.
- Each module begins with a 3–8 line **File Summary** block and exports typed APIs with Google-style docstrings (`Args`, `Returns`, `Example`).
- Keep functions short and pure where practical; prefer dependency injection for RNGs and field callables.
- Logging uses `loguru` at INFO level by default (`configure_logging` in `driftcast/__init__.py`).

## Testing

- Add unit tests under `tests/` for new logic; run `make test` (pytest + coverage).
- Simulation fixtures should use small ensembles (< 200 particles) to keep runtime under 2 minutes.
- Verify density conservation and beaching behaviour when touching `driftcast/post` or `driftcast/particles`.

## Documentation and Assets

- Update `docs/index.md` and regenerate HTML with `make docs`.
- Export figure updates to `results/figures/hero.png` (1920×1080) via the visualization utilities.
- Keep the README preview GIF under 5 MB; regenerate using the preview animation pipeline when styles change.

## Workflow

- Use feature branches named `feature/...`, `bugfix/...`, or `docs/...`.
- Follow Conventional Commit style (`feat:`, `fix:`, `docs:`, etc.).
- Open draft PRs early; request review once tests, docs, and animations (if applicable) are updated.

Thanks for helping advance open, reproducible ocean plastics forecasting!
