"""Real-time HUD and debug overlay for chess vision pipeline."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

# Unicode chess symbols for clear on-screen display
PIECE_UNICODE = {
    "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
    "p": "p", "n": "n", "b": "b", "r": "r", "q": "q", "k": "k",
}

UNICODE_GLYPHS = {
    "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
    "p": "p", "n": "n", "b": "b", "r": "r", "q": "q", "k": "k",
}


def draw_hud_overlay(
    warped_board: np.ndarray,
    predictions: Dict[str, Any],
    fps: float = 0.0,
    last_move: Optional[str] = None,
    suggested_move: Optional[str] = None,
    auto_move: bool = False,
) -> np.ndarray:
    """
    Renders an informative HUD overlay on top of the warped chess board.

    Args:
        warped_board: 512x512 BGR image.
        predictions: Output dictionary from PieceClassifier.predict_board().
        fps: Real-time frames per second.
        last_move: Optional string of last inferred move (e.g. 'e2e4').
        suggested_move: Optional engine suggested move.

    Returns:
        np.ndarray: Enhanced HUD image (512x640 with bottom info panel).
    """
    h, w = warped_board.shape[:2]
    step = w // 8

    # Create canvas: 512 board + 140px bottom dashboard
    canvas = np.zeros((h + 140, w, 3), dtype=np.uint8)
    canvas[:h, :w] = warped_board.copy()

    # Draw 8x8 grid lines
    for i in range(1, 8):
        cv2.line(canvas, (i * step, 0), (i * step, h), (0, 255, 255), 1)
        cv2.line(canvas, (0, i * step), (w, i * step), (0, 255, 255), 1)

    grid_pieces = predictions.get("grid_pieces", [])
    grid_conf = predictions.get("grid_confidences", [])
    flipped = predictions.get("flipped", False)

    # Highlight squares and annotate pieces
    for r in range(8):
        for c in range(8):
            piece = grid_pieces[r][c] if r < len(grid_pieces) and c < len(grid_pieces[r]) else None
            conf = grid_conf[r][c] if r < len(grid_conf) and c < len(grid_conf[r]) else 0.0

            x1 = c * step
            y1 = r * step

            if piece is not None and conf >= 0.25:
                # Text color: White pieces = green badge, Black pieces = cyan badge
                is_white = piece.isupper()
                badge_color = (0, 230, 0) if is_white else (255, 120, 0)
                font_scale = 0.75
                thickness = 2

                label = f"{piece}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
                text_x = x1 + (step - tw) // 2
                text_y = y1 + (step + th) // 2 - 4

                # Background badge behind letter for high contrast readability
                cv2.rectangle(
                    canvas,
                    (text_x - 4, text_y - th - 4),
                    (text_x + tw + 4, text_y + 4),
                    (20, 20, 20),
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (text_x - 4, text_y - th - 4),
                    (text_x + tw + 4, text_y + 4),
                    badge_color,
                    1,
                )
                cv2.putText(
                    canvas,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_scale,
                    badge_color,
                    thickness,
                    cv2.LINE_AA,
                )

                # Confidence label in small font
                conf_str = f"{int(conf * 100)}%"
                cv2.putText(
                    canvas,
                    conf_str,
                    (x1 + 3, y1 + step - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

    # Dashboard Area (bottom 140px)
    dashboard_y = h
    cv2.rectangle(canvas, (0, dashboard_y), (w, h + 140), (25, 25, 28), -1)
    cv2.line(canvas, (0, dashboard_y), (w, dashboard_y), (80, 80, 80), 2)

    # Status text
    fen = predictions.get("fen_placement", "Starting...")
    cv2.putText(canvas, "CHESS VISION AI - LIVE HUD", (12, dashboard_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 110, dashboard_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # FEN placement
    fen_display = fen if len(fen) <= 52 else fen[:49] + "..."
    cv2.putText(canvas, f"FEN: {fen_display}", (12, dashboard_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Move info
    orient_str = "Our side: BLACK (bottom)" if flipped else "Our side: WHITE (bottom)"
    cv2.putText(canvas, orient_str, (12, dashboard_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 100) if not flipped else (100, 200, 255), 2)

    # Auto-Move indicator
    if auto_move:
        cv2.putText(canvas, "Auto-Move: ON ('a')", (w - 200, dashboard_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    else:
        cv2.putText(canvas, "Auto-Move: OFF ('a')", (w - 200, dashboard_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.putText(canvas, "Move: 'm' | Color: 'c'", (w - 200, dashboard_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    if last_move:
        cv2.putText(canvas, f"Last Move: {last_move}", (12, dashboard_y + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

    if suggested_move:
        cv2.putText(canvas, f"Engine: {suggested_move}", (w - 200, dashboard_y + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2)

    return canvas
