"""Benchmark script: NeuralEngine vs Stockfish at multiple skill levels.

Plays N games of NeuralEngine (White) vs StockfishEngine (Black) at Skill 0, 5, 10.
Reports Win/Draw/Loss statistics.

Usage:
    python scripts/benchmark_neural.py --games 10 --skills 0 5 10 --weights models/policy_value.pt
    python scripts/benchmark_neural.py --games 5 --mcts 50
"""

from __future__ import annotations
import argparse
import os
import sys
import logging
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

STOCKFISH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "stockfish", "stockfish-windows-x86-64-avx2.exe"
)


def play_game(
    white_engine,
    black_engine,
    max_moves: int = 200,
    time_left_ms: int = 10_000,
) -> str:
    """
    Play one game between two engines.

    Returns:
        "1-0" if white wins, "0-1" if black wins, "1/2-1/2" for draw.
    """
    board = chess.Board()
    for move_num in range(max_moves):
        if board.is_game_over():
            break
        engine = white_engine if board.turn == chess.WHITE else black_engine
        try:
            move = engine.best_move(board, time_left_ms=time_left_ms)
        except Exception as e:
            logger.warning("Engine error on move %d: %s", move_num + 1, e)
            break
        if move is None or move not in board.legal_moves:
            logger.warning("Illegal/null move returned: %s. Resigning.", move)
            return "0-1" if board.turn == chess.WHITE else "1-0"
        board.push(move)
    return board.result(claim_draw=True)


def run_benchmark(
    num_games: int,
    skill_levels: List[int],
    weights_path: Optional[str],
    mcts_simulations: int,
    time_left_ms: int,
) -> None:
    """Run benchmark and print W/D/L table."""

    # Validate Stockfish availability
    if not os.path.isfile(STOCKFISH_PATH):
        logger.error(
            "Stockfish not found at %s\n"
            "Run: python scripts/download_stockfish.py",
            STOCKFISH_PATH,
        )
        sys.exit(1)

    from engine.neural_engine import NeuralEngine
    from engine.stockfish_engine import StockfishEngine

    results: Dict[int, Dict[str, int]] = {}

    for skill in skill_levels:
        logger.info(
            "=== Benchmarking NeuralEngine vs Stockfish Skill %d (%d games) ===",
            skill,
            num_games,
        )
        tally: Dict[str, int] = {"W": 0, "D": 0, "L": 0}

        neural_eng = NeuralEngine(
            weights_path=weights_path if (weights_path and os.path.isfile(weights_path)) else None,
            mcts_simulations=mcts_simulations,
            time_budget_ms=time_left_ms,
        )
        sf_eng = StockfishEngine(
            path=STOCKFISH_PATH,
            skill_level=skill,
            time_budget_ms=time_left_ms,
        )

        try:
            for g in range(num_games):
                result = play_game(
                    white_engine=neural_eng,
                    black_engine=sf_eng,
                    max_moves=200,
                    time_left_ms=time_left_ms,
                )
                if result == "1-0":
                    tally["W"] += 1
                    outcome_str = "Neural WIN"
                elif result == "0-1":
                    tally["L"] += 1
                    outcome_str = "Neural LOSS"
                else:
                    tally["D"] += 1
                    outcome_str = "DRAW"
                logger.info("  Game %d/%d: %s (%s)", g + 1, num_games, result, outcome_str)
        finally:
            neural_eng.close()
            sf_eng.close()

        results[skill] = tally

    # Print summary table
    print("\n" + "=" * 56)
    print(f"{'BENCHMARK RESULTS':^56}")
    print(f"{'NeuralEngine (White) vs Stockfish (Black)':^56}")
    print(f"{'Games per level: ' + str(num_games):^56}")
    print("=" * 56)
    print(f"{'Skill':>8}  {'W':>5}  {'D':>5}  {'L':>5}  {'Score%':>8}")
    print("-" * 56)
    for skill, tally in sorted(results.items()):
        w, d, l_ = tally["W"], tally["D"], tally["L"]
        total = w + d + l_
        score_pct = (w + 0.5 * d) / total * 100 if total > 0 else 0.0
        print(f"{skill:>8}  {w:>5}  {d:>5}  {l_:>5}  {score_pct:>7.1f}%")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NeuralEngine vs Stockfish at various skill levels."
    )
    parser.add_argument(
        "--games", type=int, default=10, help="Number of games per skill level (default: 10)"
    )
    parser.add_argument(
        "--skills",
        type=int,
        nargs="+",
        default=[0, 5, 10],
        help="Stockfish Skill Levels to benchmark against (default: 0 5 10)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/policy_value.pt",
        help="Path to trained policy/value weights (default: models/policy_value.pt)",
    )
    parser.add_argument(
        "--mcts",
        type=int,
        default=0,
        help="MCTS simulations per move (0 = greedy, default: 0)",
    )
    parser.add_argument(
        "--time-ms",
        type=int,
        default=2000,
        help="Think time per move in ms for each engine (default: 2000)",
    )
    args = parser.parse_args()

    run_benchmark(
        num_games=args.games,
        skill_levels=args.skills,
        weights_path=args.weights,
        mcts_simulations=args.mcts,
        time_left_ms=args.time_ms,
    )


if __name__ == "__main__":
    main()
