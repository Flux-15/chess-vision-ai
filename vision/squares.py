"""Splits canonical chess board into 64 square crops and manages square-to-algebraic mappings."""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2

FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"]


def get_square_name(row: int, col: int, flipped: bool = False) -> str:
    """
    Returns algebraic square name (e.g. 'e4') from (row, col) grid coordinates.

    Args:
        row: Row index 0 (top) to 7 (bottom).
        col: Column index 0 (left) to 7 (right).
        flipped: If True, board is viewed from Black's perspective (h1 at top-left).

    Returns:
        Square notation string, e.g. "e4".
    """
    if not flipped:
        file_char = FILES[col]
        rank_char = RANKS[7 - row]
    else:
        file_char = FILES[7 - col]
        rank_char = RANKS[row]
    return f"{file_char}{rank_char}"


def get_grid_coords(square_name: str, flipped: bool = False) -> Tuple[int, int]:
    """
    Returns (row, col) grid coordinates from algebraic square name (e.g. 'e4').

    Args:
        square_name: Algebraic square name (e.g., 'e4').
        flipped: If True, Black is at bottom.

    Returns:
        Tuple (row, col) in range 0..7.
    """
    file_char = square_name[0].lower()
    rank_char = square_name[1]

    file_idx = FILES.index(file_char)
    rank_idx = RANKS.index(rank_char)

    if not flipped:
        row = 7 - rank_idx
        col = file_idx
    else:
        row = rank_idx
        col = 7 - file_idx

    return row, col


class SquareSplitter:
    """Splits a 512x512 warped board into 64 square images with border margin cropping."""

    def __init__(
        self,
        board_size: int = 512,
        square_size: int = 64,
        margin_crop: int = 3,
    ) -> None:
        """
        Initialize SquareSplitter.

        Args:
            board_size: Canonical board image width/height (default 512).
            square_size: Canonical output size per square (default 64).
            margin_crop: Pixels to crop inside each square edge to avoid borders.
        """
        self.board_size = board_size
        self.square_size = square_size
        self.margin_crop = margin_crop
        self.step = board_size // 8

    def split_board(
        self,
        board_img: np.ndarray,
        flipped: bool = False,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Extracts all 64 squares from top-down board image.

        Args:
            board_img: Warped board image of shape (board_size, board_size, 3).
            flipped: True if Black is at the bottom.

        Returns:
            List of 64 tuples (square_name, square_crop) ordered rank-by-rank:
            When white at bottom: a8, b8, ..., h8, a7, ..., h1.
        """
        if board_img.shape[:2] != (self.board_size, self.board_size):
            board_img = cv2.resize(board_img, (self.board_size, self.board_size))

        squares: List[Tuple[str, np.ndarray]] = []

        for row in range(8):
            y1 = row * self.step
            y2 = (row + 1) * self.step
            for col in range(8):
                x1 = col * self.step
                x2 = (col + 1) * self.step

                sq_name = get_square_name(row, col, flipped=flipped)
                crop = board_img[y1:y2, x1:x2]

                # Inset crop to remove grid borders and piece overlaps from neighboring squares
                if self.margin_crop > 0 and self.step > 2 * self.margin_crop:
                    crop = crop[
                        self.margin_crop : self.step - self.margin_crop,
                        self.margin_crop : self.step - self.margin_crop,
                    ]

                # Resize to standard square_size (e.g. 64x64)
                if crop.shape[:2] != (self.square_size, self.square_size):
                    crop = cv2.resize(crop, (self.square_size, self.square_size), interpolation=cv2.INTER_AREA)

                squares.append((sq_name, crop))

        return squares

    def get_squares_batch(
        self,
        board_img: np.ndarray,
        flipped: bool = False,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Returns all 64 square crops packed as a single batch array (64, H, W, 3)
        along with matching list of square names.
        """
        named_squares = self.split_board(board_img, flipped=flipped)
        names = [name for name, _ in named_squares]
        crops = np.stack([crop for _, crop in named_squares], axis=0)
        return names, crops
