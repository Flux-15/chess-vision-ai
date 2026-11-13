"""Quick test of the retrained CNN + template hybrid on the saved warped board."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2, numpy as np
from vision.template_classifier import TemplateClassifier

# Expected starting position (white at bottom):
# Row 0 (rank 8): r n b q k b n r
# Row 1 (rank 7): p p p p p p p p
# Rows 2-5: empty
# Row 6 (rank 2): P P P P P P P P
# Row 7 (rank 1): R N B Q K B N R

clf = TemplateClassifier(
    ncc_threshold=0.55,
    cnn_fallback=True,
    cnn_weights="models/classifier.pt",
)

warped = cv2.imread("debug_frames/user_test_board_warp.png")
if warped is None:
    print("ERROR: debug_frames/user_test_board_warp.png not found")
    sys.exit(1)

print(f"Board image: {warped.shape}")
result = clf.predict_board(warped, flipped=False)
print(f"FEN: {result['fen_placement']}")
print()

correct_starting = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
print(f"Expected starting FEN: {correct_starting}")
print(f"Match: {result['fen_placement'] == correct_starting}")
print()

print("Grid (row=0 is rank 8):")
for r in range(8):
    pieces = " ".join(p if p else "." for p in result["grid_pieces"][r])
    confs  = " ".join(f"{c:.2f}" for c in result["grid_confidences"][r])
    print(f"  Row {r}: {pieces}  conf: {confs}")

all_confs = [result["grid_confidences"][r][c] for r in range(8) for c in range(8)]
print(f"\nMin={min(all_confs):.3f}  Max={max(all_confs):.3f}  Mean={sum(all_confs)/len(all_confs):.3f}")

# Count empty vs pieces
pieces = [p for row in result["grid_pieces"] for p in row if p is not None]
empties = [p for row in result["grid_pieces"] for p in row if p is None]
print(f"Pieces: {len(pieces)}  Empty: {len(empties)}")
print(f"Pieces found: {pieces}")
