"""Board state management, 8x8 grid to FEN conversion, and move inference via python-chess."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import chess

# Class index to FEN piece character mapping
# 0 = empty, 1..6 = White (P, N, B, R, Q, K), 7..12 = Black (p, n, b, r, q, k)
CLASS_TO_PIECE: Dict[int, Optional[str]] = {
    0: None,
    1: "P",
    2: "N",
    3: "B",
    4: "R",
    5: "Q",
    6: "K",
    7: "p",
    8: "n",
    9: "b",
    10: "r",
    11: "q",
    12: "k",
}

PIECE_TO_CLASS: Dict[Optional[str], int] = {v: k for k, v in CLASS_TO_PIECE.items()}

CLASS_NAMES: List[str] = [
    "empty",
    "wP",
    "wN",
    "wB",
    "wR",
    "wQ",
    "wK",
    "bP",
    "bN",
    "bB",
    "bR",
    "bQ",
    "bK",
]


def grid_to_fen_placement(grid: List[List[Optional[str]]], flipped: bool = False) -> str:
    """
    Converts an 8x8 grid of piece characters (or None for empty) into a FEN piece placement string.

    Args:
        grid: 8x8 list of piece characters.
              If not flipped, grid[0] is rank 8 (a8..h8) and grid[7] is rank 1 (a1..h1).
              If flipped, grid[0] is rank 1 (h1..a1) and grid[7] is rank 8 (h8..a8).
        flipped: True if captured from Black's perspective.

    Returns:
        FEN placement string, e.g. "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR".
    """
    # Normalize grid so normalized[0] is rank 8 (a8..h8) down to normalized[7] is rank 1 (a1..h1)
    if not flipped:
        normalized = grid
    else:
        # Invert ranks (0..7 -> 7..0) and reverse files (0..7 -> 7..0)
        normalized = [list(reversed(row)) for row in reversed(grid)]

    ranks_fen = []
    for row in normalized:
        empty_count = 0
        rank_str = ""
        for cell in row:
            if cell is None or cell == "" or cell == ".":
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += cell
        if empty_count > 0:
            rank_str += str(empty_count)
        ranks_fen.append(rank_str)

    return "/".join(ranks_fen)


def fen_placement_to_grid(fen_placement: str) -> List[List[Optional[str]]]:
    """
    Converts a FEN placement string into a normalized 8x8 grid (rank 8 at index 0, rank 1 at 7).
    """
    ranks = fen_placement.split()[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN placement ranks count: {len(ranks)}")

    grid: List[List[Optional[str]]] = []
    for r in ranks:
        row: List[Optional[str]] = []
        for ch in r:
            if ch.isdigit():
                row.extend([None] * int(ch))
            else:
                row.append(ch)
        if len(row) != 8:
            raise ValueError(f"Rank {r} did not parse to 8 squares (got {len(row)})")
        grid.append(row)
    return grid


class GameStateManager:
    """Maintains full chess game state and infers legal moves between frames."""

    def __init__(self, initial_fen: str = chess.STARTING_FEN) -> None:
        """
        Initialize game state with starting FEN.

        Args:
            initial_fen: Starting FEN string.
        """
        self.board = chess.Board(initial_fen)
        self.move_history: List[chess.Move] = []
        self.last_detected_fen: Optional[str] = None
        self.unstable_frames: int = 0

    def get_fen(self) -> str:
        """Returns current full FEN string."""
        return self.board.fen()

    def get_piece_placement(self) -> str:
        """Returns board piece placement portion of current FEN."""
        return self.board.board_fen()

    def count_placement_differences(self, placement_a: str, placement_b: str) -> int:
        """Counts how many of the 64 squares differ between two FEN placements."""
        grid_a = fen_placement_to_grid(placement_a)
        grid_b = fen_placement_to_grid(placement_b)
        diffs = 0
        for r in range(8):
            for c in range(8):
                if grid_a[r][c] != grid_b[r][c]:
                    diffs += 1
        return diffs

    def infer_move_from_detected_grid(
        self,
        detected_grid: List[List[Optional[str]]],
        flipped: bool = False,
    ) -> Tuple[Optional[chess.Move], float]:
        """
        Matches a detected 8x8 piece grid against legal moves in the current position.

        Args:
            detected_grid: 8x8 grid of piece characters (or None).
            flipped: True if Black is at the bottom.

        Returns:
            Tuple of (best_legal_move, confidence_score_0_to_1).
            If no legal move matches or position is unchanged, move will be None.
        """
        detected_placement = grid_to_fen_placement(detected_grid, flipped=flipped)
        current_placement = self.get_piece_placement()

        # If detected position is identical to current position, no move occurred
        if detected_placement == current_placement:
            return None, 1.0

        best_move: Optional[chess.Move] = None
        best_diff = 64
        exact_matches: List[chess.Move] = []

        # Test each legal move against the detected placement
        for move in self.board.legal_moves:
            test_board = self.board.copy()
            test_board.push(move)
            test_placement = test_board.board_fen()

            diff = self.count_placement_differences(test_placement, detected_placement)

            if diff == 0:
                exact_matches.append(move)
            elif diff < best_diff:
                best_diff = diff
                best_move = move

        # Exact match found (preferred)
        if len(exact_matches) == 1:
            return exact_matches[0], 1.0
        elif len(exact_matches) > 1:
            # Ambiguous (rare, e.g. queen promotion choice when piece model can't distinguish)
            # Default to queen promotion if among them
            for m in exact_matches:
                if m.promotion == chess.QUEEN:
                    return m, 0.95
            return exact_matches[0], 0.9

        # In case of small vision noise (e.g. 1 square glitch), allow match with diff <= 2
        # if diff from current is significant (e.g. pieces actually moved)
        diff_from_current = self.count_placement_differences(current_placement, detected_placement)
        if best_move is not None and best_diff <= 2 and diff_from_current >= 2:
            confidence = max(0.5, (64 - best_diff) / 64.0)
            return best_move, confidence

        return None, 0.0

    def apply_move(self, move: chess.Move) -> bool:
        """
        Pushes a legal move to advance the game state.

        Args:
            move: python-chess Move object.

        Returns:
            True if move was legal and applied, False otherwise.
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            self.last_detected_fen = self.board.fen()
            return True
        return False

    def resync(self, fen: str) -> bool:
        """
        Resync game state to an explicit FEN if tracking drifts.

        Args:
            fen: Target FEN string.

        Returns:
            True if FEN was valid and loaded, False otherwise.
        """
        try:
            new_board = chess.Board(fen)
            self.board = new_board
            self.last_detected_fen = fen
            return True
        except ValueError:
            return False

    def resync_from_fen(self, fen: str) -> bool:
        """Alias for resync for backward compatibility."""
        return self.resync(fen)

    def is_game_over(self) -> bool:
        """Check if game has concluded (checkmate, stalemate, repetition, etc.)."""
        return self.board.is_game_over()
