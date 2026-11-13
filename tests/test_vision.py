"""Tests for vision pipeline: board detection, perspective warping, square splitting, and synthetic rendering."""

from __future__ import annotations
import numpy as np
import pytest
from vision.board_detect import order_quad_points, BoardDetector
from vision.squares import get_square_name, get_grid_coords, SquareSplitter
from data.generate_dataset import draw_piece_sprite, BoardRenderer, THEMES, PIECE_STYLES


def test_order_quad_points() -> None:
    """Verifies that arbitrary permutation of 4 points orders correctly."""
    # (top-left, top-right, bottom-right, bottom-left)
    true_corners = np.array([[10, 10], [100, 15], [105, 110], [12, 102]], dtype=np.float32)

    # Scramble corners
    scrambled = true_corners[[2, 0, 3, 1]]
    ordered = order_quad_points(scrambled)

    # Verify top-left has smallest sum
    assert np.allclose(ordered[0], true_corners[0])
    # Verify bottom-right has largest sum
    assert np.allclose(ordered[2], true_corners[2])


def test_square_name_and_coords_roundtrip() -> None:
    """Tests square name to grid coordinate mappings across all 64 squares."""
    for flipped in [False, True]:
        for r in range(8):
            for c in range(8):
                sq_name = get_square_name(r, c, flipped=flipped)
                row_back, col_back = get_grid_coords(sq_name, flipped=flipped)
                assert (row_back, col_back) == (r, c)


def test_square_splitter() -> None:
    """Tests splitting a 512x512 board into 64 64x64 square crops."""
    dummy_board = np.zeros((512, 512, 3), dtype=np.uint8)
    splitter = SquareSplitter(board_size=512, square_size=64, margin_crop=3)

    squares = splitter.split_board(dummy_board, flipped=False)
    assert len(squares) == 64

    # Check first and last square names
    assert squares[0][0] == "a8"
    assert squares[63][0] == "h1"

    # Check crop shapes
    for name, crop in squares:
        assert crop.shape == (64, 64, 3)

    # Check batch extraction
    names, batch = splitter.get_squares_batch(dummy_board)
    assert len(names) == 64
    assert batch.shape == (64, 64, 64, 3)


def test_piece_sprite_rendering() -> None:
    """Tests rendering piece sprites across styles and types."""
    for piece in ["P", "N", "B", "R", "Q", "K"]:
        for color in ["w", "b"]:
            for style in PIECE_STYLES:
                sprite = draw_piece_sprite(piece, color, style=style, size=64)
                assert sprite.size == (64, 64)
                assert sprite.mode == "RGBA"


def test_board_renderer_and_augmentations() -> None:
    """Tests full board rendering with themes and highlight overlays."""
    renderer = BoardRenderer(target_square_size=64)
    grid = [["wK" if (r == 7 and c == 4) else None for c in range(8)] for r in range(8)]
    grid[0][4] = "bK"

    for theme_name in THEMES.keys():
        board_bgr, labels = renderer.render_position(
            placement_grid=grid,
            theme_name=theme_name,
            piece_style="cburnett",
            last_move_squares=[(6, 4), (4, 4)],
            check_square=(0, 4),
            add_noise=False,
        )

        assert board_bgr.shape == (512, 512, 3)
        assert len(labels) == 64
