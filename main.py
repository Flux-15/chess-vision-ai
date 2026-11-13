"""Main entrypoint for Vision-based Chess AI.

Launches the interactive setup GUI or direct CLI pipelines.

Modes:
  (default)          → Interactive Tkinter setup launcher
  --train            → Train the 13-class square classifier (Phase 1)
  --train-policy     → Train the policy/value network on Lichess PGN data (Phase 3)
  --test             → Single-frame board detection and FEN output
  --live / --cli     → Continuous live vision + HUD loop (local_test_mode)
  --lichess          → Launch Lichess Bot API loop (Phase 4)
  --benchmark        → Benchmark NeuralEngine vs Stockfish
"""

from __future__ import annotations
import sys
import argparse
import logging


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vision-based Chess AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Vision / general ----
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and save frames")
    parser.add_argument("--cli", action="store_true", help="Headless mode (skip GUI, run live loop)")
    parser.add_argument("--live", action="store_true", help="Launch live vision capture and HUD loop")
    parser.add_argument("--auto-move", action="store_true", help="Enable automatic move clicking with pyautogui in vs-computer mode")
    parser.add_argument("--test", action="store_true", help="Capture and calibrate a single screen frame")

    # ---- Classifier training (Phase 1) ----
    parser.add_argument("--train", action="store_true", help="Train the 13-class square classifier")
    parser.add_argument("--train-samples", type=int, default=80000, help="Training square samples (default: 80000)")
    parser.add_argument("--val-samples", type=int, default=20000, help="Validation square samples (default: 20000)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")

    # ---- Policy training (Phase 3) ----
    parser.add_argument("--train-policy", action="store_true", help="Train policy/value network on Lichess PGN data")
    parser.add_argument("--pgn-dir", type=str, default="data/pgn", help="Directory of Lichess PGN files (default: data/pgn)")
    parser.add_argument("--policy-epochs", type=int, default=10, help="Policy training epochs (default: 10)")
    parser.add_argument("--policy-batch", type=int, default=512, help="Policy training batch size (default: 512)")
    parser.add_argument("--min-elo", type=int, default=1800, help="Minimum player Elo filter for PGN (default: 1800)")
    parser.add_argument("--policy-output", type=str, default="models/policy_value.pt", help="Output model path")

    # ---- Lichess Bot (Phase 4) ----
    parser.add_argument("--lichess", action="store_true", help="Launch Lichess Bot API loop")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["stockfish", "neural"],
        default=None,
        help="Engine to use for Lichess bot (overrides config.yaml)",
    )
    parser.add_argument("--skill", type=int, default=None, help="Stockfish skill level 0-20 (overrides config.yaml)")
    parser.add_argument("--mcts", type=int, default=None, help="MCTS simulations for neural engine (overrides config.yaml)")

    # ---- Benchmark (Phase 3) ----
    parser.add_argument("--benchmark", action="store_true", help="Benchmark NeuralEngine vs Stockfish")
    parser.add_argument("--bench-games", type=int, default=10, help="Games per skill level for benchmark (default: 10)")
    parser.add_argument("--bench-skills", type=int, nargs="+", default=[0, 5, 10], help="Stockfish skill levels to test (default: 0 5 10)")
    parser.add_argument("--bench-weights", type=str, default="models/policy_value.pt", help="Neural weights for benchmark")

    args = parser.parse_args()
    _setup_logging(args.debug)

    # ------------------------------------------------------------------ #
    # Phase 1 – Train classifier
    # ------------------------------------------------------------------ #
    if args.train:
        from training.train_classifier import train_classifier
        train_classifier(
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        return

    # ------------------------------------------------------------------ #
    # Phase 3 – Train policy/value network
    # ------------------------------------------------------------------ #
    if args.train_policy:
        from training.train_policy import train_policy
        train_policy(
            pgn_dir=args.pgn_dir,
            epochs=args.policy_epochs,
            batch_size=args.policy_batch,
            min_elo=args.min_elo,
            output_path=args.policy_output,
        )
        return

    # ------------------------------------------------------------------ #
    # Phase 3 – Benchmark neural vs Stockfish
    # ------------------------------------------------------------------ #
    if args.benchmark:
        from scripts.benchmark_neural import run_benchmark
        run_benchmark(
            num_games=args.bench_games,
            skill_levels=args.bench_skills,
            weights_path=args.bench_weights,
            mcts_simulations=args.mcts or 0,
            time_left_ms=2000,
        )
        return

    # ------------------------------------------------------------------ #
    # Phase 4 – Lichess Bot
    # ------------------------------------------------------------------ #
    if args.lichess:
        import yaml, os
        config_path = "config.yaml"
        cfg: dict = {}
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}

        engine_type = args.engine or cfg.get("engine", {}).get("type", "stockfish")

        # CLI overrides
        if args.skill is not None:
            cfg.setdefault("engine", {})["skill_level"] = args.skill
        if args.mcts is not None:
            cfg.setdefault("engine", {})["mcts_simulations"] = args.mcts

        from play.lichess_bot import LichessBot
        bot = LichessBot(engine_type=engine_type, config=cfg)
        bot.run()
        return

    # ------------------------------------------------------------------ #
    # Phase 2 – Vision: single-frame test
    # ------------------------------------------------------------------ #
    if args.test:
        from play.local_test_mode import run_single_frame_test
        run_single_frame_test()
        return

    # ------------------------------------------------------------------ #
    # Phase 2 – Vision: live loop (CLI/headless)
    # ------------------------------------------------------------------ #
    if args.live or args.cli:
        from play.local_test_mode import run_live_loop
        run_live_loop(auto_move=args.auto_move)
        return

    # ------------------------------------------------------------------ #
    # Default – Interactive Tkinter setup GUI
    # ------------------------------------------------------------------ #
    from play.launcher_gui import launch_gui
    launch_gui()


if __name__ == "__main__":
    main()
