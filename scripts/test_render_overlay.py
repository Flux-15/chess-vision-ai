"""Generates a sample visual HUD image to verify debug overlay formatting."""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import chess
from vision.classifier import PieceClassifier
from vision.overlay import draw_hud_overlay
from data.generate_dataset import BoardRenderer

WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pt"))


def generate_sample_hud() -> str:
    classifier = PieceClassifier(weights_path=WEIGHTS_PATH)
    renderer = BoardRenderer(target_square_size=64)

    # Standard starting board with 1. e4 highlight simulation
    grid = [
        ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
        ["bP"] * 8,
        [None] * 8,
        [None] * 8,
        [None, None, None, None, "wP", None, None, None],
        [None] * 8,
        ["wP", "wP", "wP", "wP", None, "wP", "wP", "wP"],
        ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
    ]

    board_bgr, _ = renderer.render_position(
        placement_grid=grid,
        theme_name="green",
        piece_style="cburnett",
        last_move_squares=[(6, 4), (4, 4)],  # e2-e4
        check_square=None,
        add_noise=False,
    )

    preds = classifier.predict_board(board_bgr, flipped=False)
    hud = draw_hud_overlay(
        board_bgr,
        predictions=preds,
        fps=64.2,
        last_move="e2e4",
        suggested_move="e7e5",
    )

    os.makedirs("debug_frames", exist_ok=True)
    out_path = os.path.abspath("debug_frames/sample_hud.png")
    cv2.imwrite(out_path, hud)
    print(f"[*] Sample HUD generated and saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_sample_hud()
