"""Integration tests for PieceClassifier using trained model weights."""

from __future__ import annotations
import os
import pytest
import numpy as np
import chess
from vision.classifier import PieceClassifier
from data.generate_dataset import BoardRenderer, THEMES, PIECE_STYLES

WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pt"))


@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Trained weights not found")
def test_classifier_end_to_end_starting_position() -> None:
    """Tests that full board starting position renders, classifies, and outputs exact starting FEN."""
    classifier = PieceClassifier(weights_path=WEIGHTS_PATH)

    # Standard starting board layout
    grid = [
        ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
        ["bP"] * 8,
        [None] * 8,
        [None] * 8,
        [None] * 8,
        [None] * 8,
        ["wP"] * 8,
        ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
    ]

    renderer = BoardRenderer(target_square_size=64)
    board_bgr, _ = renderer.render_position(
        placement_grid=grid,
        theme_name="green",
        piece_style="cburnett",
        last_move_squares=[(6, 4), (4, 4)],  # e2-e4 highlight simulation
        check_square=None,
        add_noise=False,
    )

    preds = classifier.predict_board(board_bgr, flipped=False)
    expected_fen = chess.STARTING_BOARD_FEN  # "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    assert preds["fen_placement"] == expected_fen
    assert preds["flipped"] is False


@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Trained weights not found")
def test_classifier_held_out_generalization() -> None:
    """Tests end-to-end classification on held-out 'icy' theme and 'alpha' piece set."""
    classifier = PieceClassifier(weights_path=WEIGHTS_PATH)

    # Custom middlegame position
    grid = [[None for _ in range(8)] for _ in range(8)]
    grid[0][4] = "bK"
    grid[1][4] = "bP"
    grid[6][4] = "wP"
    grid[7][4] = "wK"

    renderer = BoardRenderer(target_square_size=64)
    board_bgr, _ = renderer.render_position(
        placement_grid=grid,
        theme_name="icy",       # Held-out theme
        piece_style="alpha",    # Held-out piece style
        last_move_squares=[(6, 4), (4, 4)],
        check_square=None,
        add_noise=True,
    )

    preds = classifier.predict_board(board_bgr, flipped=False)
    expected_fen = "4k3/4p3/8/8/8/8/4P3/4K3"

    assert preds["fen_placement"] == expected_fen
