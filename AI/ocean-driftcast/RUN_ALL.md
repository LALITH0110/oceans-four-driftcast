# Driftcast Runbook (Conda + Windows PowerShell)

Follow the commands below in order from the repository root. Each block is copy-pastable into PowerShell.

## 0. Install Conda (Miniforge) if missing
Run this once. It installs Miniforge into `%USERPROFILE%\miniforge3`, adds it to PATH, and initializes PowerShell. Restart the terminal afterwards.
```powershell
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    $installer = "$env:TEMP\Miniforge3-latest-Windows-x86_64.exe"
    Invoke-WebRequest -Uri https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe -OutFile $installer
    Start-Process -FilePath $installer -Wait -ArgumentList "/InstallationType=JustMe","/AddToPath=1","/RegisterPython=0","/S","/D=$env:USERPROFILE\miniforge3"
    & "$env:USERPROFILE\miniforge3\Scripts\conda.exe" init powershell
    Write-Host "`nMiniforge installed. Close this PowerShell window and open a new one before continuing." -ForegroundColor Yellow
    exit
} else {
    Write-Host "Conda already available. Continue to Step 1."
}
```

## 1. Create/refresh the Conda environment
```powershell
conda env create -f environment.yml
```
If the environment already exists, refresh it instead:
```powershell
conda env update -f environment.yml --prune
```

## 2. Activate the environment
```powershell
conda activate driftcast
```

## 3. Ensure setuptools only discovers the `driftcast` package
```powershell
$script = @"
from pathlib import Path

path = Path('pyproject.toml')
snippet = '[tool.setuptools.packages.find]\ninclude = ["driftcast*"]\n'
text = path.read_text(encoding='utf8')
if '[tool.setuptools.packages.find]' not in text:
    path.write_text(text.rstrip() + '\n\n' + snippet, encoding='utf8')
    print('Added setuptools package discovery snippet.')
else:
    print('Package discovery snippet already present. No changes made.')
"@
python -c $script
```

## 4. Install Driftcast in editable mode
```powershell
pip install -e .
```

## 5. (Optional) Install pre-commit hooks
```powershell
make precommit
```

## 6. Verify FFmpeg availability
```powershell
python -c "from driftcast.viz.ffmpeg import detect_ffmpeg; print(detect_ffmpeg())"
```
If this command fails, install FFmpeg (e.g., `scoop install ffmpeg`) and rerun it.

## 7. Generate synthetic demo data
```powershell
python scripts/generate_synthetic_inputs.py
python -m driftcast.cli ingest normalize data/raw/mock_crowd.json --out-dir data/crowd/processed
```

## 8. Run the baseline simulation
```powershell
python -m driftcast.cli run configs/natl_subtropical_gyre.yaml --seed 42
```

## 9. Render preview and final animations
```powershell
python -m driftcast.cli animate preview configs/natl_coastal.yaml --seed 42
python -m driftcast.cli animate final configs/natl_coastal.yaml --seed 42
```

## 10. Build the full judge deliverables bundle
```powershell
python -m driftcast.cli judge --config configs/natl_subtropical_gyre.yaml --seed 42
```

## 11. Inspect generated artifacts
- Videos: `results/videos/preview.mp4`, `results/videos/final_cut.mp4`
- Hero frame: `results/figures/hero.png`
- Simulation outputs: `results/outputs/` (NetCDF files with manifest sidecars)
- Judge PDF: `docs/onepager.pdf`

## 12. Optional diagnostics
```powershell
python -m driftcast.cli perf check --config configs/natl_subtropical_gyre.yaml --seed 123
python -m driftcast.cli sweep --config configs/natl_shipping.yaml --param physics.diffusivity_m2s=5,10 --seed 42
```

## 13. Deactivate the environment when finished
```powershell
conda deactivate
```
