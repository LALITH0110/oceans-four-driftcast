# # DriftCast command-line interface.
# # Exposes data preparation, simulation, training, evaluation, and forecasting subcommands.
# # Acts as the single entry point for researchers running the end-to-end workflow.

# from __future__ import annotations

# import argparse
# import logging
# from datetime import datetime
# from pathlib import Path
# from typing import Optional

# from stable_baselines3 import PPO

# from src.config import DataConfig
# from src.error_utils import predict_location
# from src.rl_drift_correction import evaluate_correction
# from src.train_pipeline import (
#     build_uncertainty_plot,
#     orchestrate,
#     preprocess_data,
#     run_baseline,
#     train_task_a,
#     train_task_b,
# )

# LOGGER = logging.getLogger("driftcast")


# def _configure_logging(verbose: bool) -> None:
#     level = logging.DEBUG if verbose else logging.INFO
#     logging.basicConfig(level=level, format="%(asctime)s %(levelname)s | %(message)s")


# def cmd_prepare(args: argparse.Namespace) -> None:
#     cfg = DataConfig()
#     print(preprocess_data(cfg))


# def cmd_simulate(args: argparse.Namespace) -> None:
#     cfg = DataConfig(dt_hours=args.dt_hours, n_particles_default=args.n_particles)
#     plot = run_baseline(cfg, hours=args.hours, n_particles=args.n_particles)
#     print(f"Saved baseline plot to {plot}")


# def cmd_train_a(args: argparse.Namespace) -> None:
#     cfg = DataConfig()
#     model_path = train_task_a(cfg, timesteps=int(args.timesteps))
#     print(f"Task A model saved to {model_path}")


# def cmd_eval_a(args: argparse.Namespace) -> None:
#     cfg = DataConfig()
#     model_path = Path(args.model or "models/correction.zip")
#     if not model_path.exists():
#         raise FileNotFoundError(f"Correction model not found at {model_path}")
#     model = PPO.load(model_path)
#     metrics = evaluate_correction(model, episodes=args.episodes, cfg=cfg)
#     print(f"RMSE: {metrics['rmse']:.3f} degrees")


# def cmd_train_b(args: argparse.Namespace) -> None:
#     cfg = DataConfig()
#     model_path = train_task_b(cfg, episodes=args.episodes)
#     print(f"Task B model saved to {model_path}")


# def cmd_predict(args: argparse.Namespace) -> None:
#     cfg = DataConfig()
#     lat, lon = [float(v) for v in args.pos.split(",")]
#     message = predict_location(cfg, (lat, lon), datetime.fromisoformat(args.time), args.hours)
#     print(message)
#     plot = build_uncertainty_plot(cfg, hours_ahead=args.hours)
#     print(f"Saved uncertainty plot to {plot}")


# def cmd_train_pipeline(args: argparse.Namespace) -> None:
#     cfg = DataConfig()
#     artefacts = orchestrate(cfg=cfg, timesteps_a=int(args.timesteps_a), episodes_b=int(args.episodes_b))
#     for name, path in artefacts.items():
#         print(f"{name}: {path}")


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(prog="driftcast", description="North Atlantic plastic drift RL sandbox.")
#     parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
#     sub = parser.add_subparsers(dest="command", required=True)

#     prepare = sub.add_parser("prepare", help="Scan local datasets and print availability.")
#     prepare.set_defaults(func=cmd_prepare)

#     simulate = sub.add_parser("simulate", help="Run a short baseline simulation and save plot(s).")
#     simulate.add_argument("--hours", type=int, default=24)
#     simulate.add_argument("--n-particles", type=int, default=2000)
#     simulate.add_argument("--dt-hours", type=float, default=1.0)
#     simulate.set_defaults(func=cmd_simulate)

#     train_a = sub.add_parser("train-a", help="Train Task A drift-correction agent.")
#     train_a.add_argument("--timesteps", type=float, default=1e5)
#     train_a.set_defaults(func=cmd_train_a)

#     eval_a = sub.add_parser("eval-a", help="Evaluate Task A correction model.")
#     eval_a.add_argument("--model", type=str, default="models/correction.zip")
#     eval_a.add_argument("--episodes", type=int, default=5)
#     eval_a.set_defaults(func=cmd_eval_a)

#     train_b = sub.add_parser("train-b", help="Train Task B cleanup agent.")
#     train_b.add_argument("--episodes", type=int, default=300)
#     train_b.set_defaults(func=cmd_train_b)

#     predict = sub.add_parser("predict", help="Run ensemble forecast with uncertainty bounds.")
#     predict.add_argument("--time", type=str, required=True)
#     predict.add_argument("--pos", type=str, required=True, help="lat,lon degrees")
#     predict.add_argument("--hours", type=int, default=24)
#     predict.set_defaults(func=cmd_predict)

#     pipeline = sub.add_parser("train-all", help="Train Tasks A and B sequentially.")
#     pipeline.add_argument("--timesteps-a", type=float, default=1e5)
#     pipeline.add_argument("--episodes-b", type=int, default=300)
#     pipeline.set_defaults(func=cmd_train_pipeline)

#     return parser


# def main(argv: Optional[list] = None) -> None:
#     parser = build_parser()
#     args = parser.parse_args(argv)
#     _configure_logging(args.verbose)
#     args.func(args)


# if __name__ == "__main__":
#     main()


# Minimal DriftCast CLI using only CMEMS currents, ERA5 winds, and GDP drifters CSV.
# Commands:
#   python main.py prepare
#   python main.py simulate --hours 24 --n-particles 1000
#   python main.py forecast --start "2023-01-15T00:00:00" --pos "40.0,-40.0" --hours 24 --runs 100

from __future__ import annotations
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.config import Config
from src.data import (
    load_cmems_currents,
    load_era5_winds,
    load_drifters_csv,
)
from src.sim import init_particles, run_simulation
from src.forecast import run_ensemble_forecast
from src.viz import plot_trajectories, plot_prob_heatmap

LOGGER = logging.getLogger("driftmini")

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s | %(message)s")

def cmd_prepare(args: argparse.Namespace) -> None:
    cfg = Config()
    cur = load_cmems_currents(cfg)
    wnd = load_era5_winds(cfg)
    drf, info = load_drifters_csv(cfg)
    print("Data summary:")
    print(f"  Currents file(s): {info['currents_files']}")
    print(f"  Winds file(s):    {info['winds_files']}")
    print(f"  Drifters rows:    {len(drf)} (after clean)")
    print(f"  Duplicates removed: {info['dups_removed']}")
    print(f"  Time span (drifters): {drf['time'].min()} to {drf['time'].max()}" if not drf.empty else "  No drifter times")

def cmd_simulate(args: argparse.Namespace) -> None:
    cfg = Config()
    currents = load_cmems_currents(cfg)
    winds = load_era5_winds(cfg)
    if currents is None or winds is None:
        raise RuntimeError("Missing currents or winds files under data/cmems or data/era5")
    state = init_particles(cfg, n=args.n_particles)
    start = datetime.utcnow()
    end = start + timedelta(hours=args.hours)
    traj = run_simulation(cfg, currents, winds, start, end, state)
    out = Path("outputs/baseline_trajectories.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_trajectories(traj["lat"], traj["lon"], out, cfg.bbox)
    print(f"Saved {out}")

def cmd_forecast(args: argparse.Namespace) -> None:
    cfg = Config()
    currents = load_cmems_currents(cfg)
    winds = load_era5_winds(cfg)
    if currents is None or winds is None:
        raise RuntimeError("Missing currents or winds files under data/cmems or data/era5")
    lat, lon = [float(x) for x in args.pos.split(",")]
    start = datetime.fromisoformat(args.start)
    end = start + timedelta(hours=args.hours)
    results = run_ensemble_forecast(
        cfg=cfg,
        currents=currents,
        winds=winds,
        start=start,
        end=end,
        start_lat=lat,
        start_lon=lon,
        n_particles=args.n_particles,
        runs=args.runs,
        perturbation_scale=args.perturb,
    )
    # Save outputs
    Path("outputs").mkdir(parents=True, exist_ok=True)
    plot_trajectories(results["lat_mean"], results["lon_mean"], Path("outputs/forecast_tracks.png"), cfg.bbox)
    plot_prob_heatmap(results["final_lat_all"], results["final_lon_all"], Path("outputs/prob_heatmap.png"), cfg.bbox)
    results["csv"].to_csv("outputs/ensemble_positions.csv", index=False)
    # Print 10:10 message if within window, else print end time
    target = start.replace(hour=10, minute=10, second=0, microsecond=0)
    if not (start <= target <= end):
        target = end
    msg = results["message_fn"](target)
    print(msg)
    print("Saved outputs to outputs/")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftmini", description="One-day plastic drift simulation and forecast")
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("prepare", help="Show what data is available and clean drifters")
    a.set_defaults(func=cmd_prepare)

    s = sub.add_parser("simulate", help="Run a baseline simulation and save a plot")
    s.add_argument("--hours", type=int, default=24)
    s.add_argument("--n-particles", type=int, default=1000)
    s.set_defaults(func=cmd_simulate)

    f = sub.add_parser("forecast", help="Run ensemble forecast with uncertainty")
    f.add_argument("--start", type=str, required=True, help='ISO time, e.g. "2023-01-15T00:00:00"')
    f.add_argument("--pos", type=str, required=True, help="lat,lon")
    f.add_argument("--hours", type=int, default=24)
    f.add_argument("--n-particles", type=int, default=100)   # per run
    f.add_argument("--runs", type=int, default=50)           # ensemble size
    f.add_argument("--perturb", type=float, default=0.1, help="scale for windage and diffusion perturbations")
    f.set_defaults(func=cmd_forecast)
    return p

def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    args.func(args)

if __name__ == "__main__":
    main()
