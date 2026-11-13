"""Test classifier on a synthetically rendered starting position."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2, numpy as np
from data.generate_dataset import FastBoardRenderer, THEMES
from vision.template_classifier import TemplateClassifier

# Render starting position on ocean theme
renderer = FastBoardRenderer(target_square_size=64)
theme = THEMES["ocean_real"]

start_grid = [
    ["bR","bN","bB","bQ","bK","bB","bN","bR"],
    ["bP","bP","bP","bP","bP","bP","bP","bP"],
    [None]*8, [None]*8, [None]*8, [None]*8,
    ["wP","wP","wP","wP","wP","wP","wP","wP"],
    ["wR","wN","wB","wQ","wK","wB","wN","wR"],
]

board_img = np.zeros((512, 512, 3), dtype=np.uint8)
for r in range(8):
    for c in range(8):
        sq = renderer.render_square(
            start_grid[r][c], (r+c)%2==0, theme, style="glass", add_noise=False
        )
        board_img[r*64:(r+1)*64, c*64:(c+1)*64] = sq

cv2.imwrite("debug_frames/synth_start.png", board_img)
print("Synthetic board saved -> debug_frames/synth_start.png")

clf = TemplateClassifier(cnn_weights="models/classifier.pt")
result = clf.predict_board(board_img, flipped=False)

expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
fen = result["fen_placement"]
print(f"Got FEN:      {fen}")
print(f"Expected FEN: {expected}")
print(f"MATCH: {fen == expected}")

confs = [result["grid_confidences"][r][c] for r in range(8) for c in range(8)]
print(f"Confidence: min={min(confs):.3f} mean={sum(confs)/64:.3f} max={max(confs):.3f}")

print("\nPer-row detail:")
for r in range(8):
    pieces = " ".join(p if p else "." for p in result["grid_pieces"][r])
    confs_r = " ".join(f"{result['grid_confidences'][r][c]:.2f}" for c in range(8))
    print(f"  Row {r}: {pieces}  conf: {confs_r}")
