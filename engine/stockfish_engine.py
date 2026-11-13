"""Stockfish UCI engine wrapper with configurable strength and time management."""

from __future__ import annotations
import os
import sys
import logging
from typing import Optional, Dict, Any
import chess
import chess.engine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.base import BaseEngine

logger = logging.getLogger(__name__)


class StockfishEngine(BaseEngine):
    """Wraps Stockfish via python-chess UCI protocol."""

    DEFAULT_PATH_WINDOWS = "stockfish/stockfish-windows-x86-64-avx2.exe"

    def __init__(
        self,
        path: Optional[str] = None,
        skill_level: int = 10,
        uci_elo: Optional[int] = None,
        time_budget_ms: int = 1000,
        threads: int = 2,
        hash_mb: int = 128,
    ) -> None:
        """
        Initialize Stockfish engine.

        Args:
            path: Path to the Stockfish executable. Auto-detected if None.
            skill_level: Stockfish Skill Level (0-20). Overridden by uci_elo if set.
            uci_elo: Target UCI_Elo rating (1350-3190). Takes precedence over skill_level.
            time_budget_ms: Default think time per move in milliseconds.
            threads: Number of CPU threads Stockfish may use.
            hash_mb: Hash table size in megabytes.
        """
        self.path = path or self._find_executable()
        self.skill_level = skill_level
        self.uci_elo = uci_elo
        self.time_budget_ms = time_budget_ms
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._start()

    def _find_executable(self) -> str:
        """Auto-detects Stockfish binary in known locations."""
        candidates = [
            self.DEFAULT_PATH_WINDOWS,
            os.path.join(os.path.dirname(__file__), "..", self.DEFAULT_PATH_WINDOWS),
        ]
        # Also search PATH
        import shutil
        sf_path = shutil.which("stockfish")
        if sf_path:
            return sf_path

        for c in candidates:
            abs_c = os.path.abspath(c)
            if os.path.isfile(abs_c):
                return abs_c

        raise FileNotFoundError(
            "Stockfish executable not found. Please download it from https://stockfishchess.org/download/ "
            "and place it at stockfish/stockfish-windows-x86-64-avx2.exe, or run: "
            "python scripts/download_stockfish.py"
        )

    def _start(self) -> None:
        """Launches Stockfish process and applies UCI options."""
        logger.info(f"[Stockfish] Starting engine: {self.path}")
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)

        options: Dict[str, Any] = {
            "Threads": self.threads,
            "Hash": self.hash_mb,
        }

        if self.uci_elo is not None:
            options["UCI_LimitStrength"] = True
            options["UCI_Elo"] = max(1350, min(3190, self.uci_elo))
            logger.info(f"[Stockfish] UCI_Elo limited to {options['UCI_Elo']}")
        else:
            options["Skill Level"] = max(0, min(20, self.skill_level))
            logger.info(f"[Stockfish] Skill Level set to {self.skill_level}")

        self._engine.configure(options)

    def best_move(self, board: chess.Board, time_left_ms: int = 1000) -> Optional[chess.Move]:
        """Returns best move from Stockfish with dynamic time budgeting."""
        if board.is_game_over():
            return None

        if self._engine is None:
            self._start()

        # Dynamic time allocation: use min of configured budget and actual time left
        think_ms = min(self.time_budget_ms, max(50, time_left_ms // 30))

        try:
            result = self._engine.play(
                board,
                chess.engine.Limit(time=think_ms / 1000.0),
            )
            return result.move
        except chess.engine.EngineTerminatedError:
            logger.error("[Stockfish] Engine terminated unexpectedly. Restarting...")
            self._start()
            return None

    def analyse(self, board: chess.Board, depth: int = 18) -> Dict[str, Any]:
        """
        Returns centipawn evaluation and principal variation for the current position.

        Args:
            board: Position to analyse.
            depth: Search depth (default 18 plies).

        Returns:
            Dict with keys: score_cp (int), pv (list of UCI moves), mate (int or None).
        """
        if self._engine is None:
            return {}

        info = self._engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info.get("score", chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
        pov_score = score.white()

        result = {
            "score_cp": pov_score.score(mate_score=10000) or 0,
            "mate": pov_score.mate(),
            "pv": [m.uci() for m in info.get("pv", [])[:5]],
            "depth": info.get("depth", 0),
        }
        return result

    def close(self) -> None:
        """Cleanly terminates the Stockfish process."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
            logger.info("[Stockfish] Engine closed.")
