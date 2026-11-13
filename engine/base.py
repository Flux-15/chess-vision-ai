"""Abstract base class for all chess engine backends."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import chess


class BaseEngine(ABC):
    """Common interface that every engine backend must implement."""

    @abstractmethod
    def best_move(self, board: chess.Board, time_left_ms: int = 1000) -> Optional[chess.Move]:
        """
        Returns the best legal move for the current position.

        Args:
            board: Current python-chess Board state.
            time_left_ms: Approximate remaining clock time in milliseconds.

        Returns:
            A legal chess.Move, or None if the game is over.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the engine."""

    def __enter__(self) -> "BaseEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
