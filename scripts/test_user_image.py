import cv2
import sys
sys.path.insert(0, ".")
from vision.classifier import PieceClassifier

img = cv2.imread("debug_frames/user_test_board_warp.png")
pc = PieceClassifier()
res = pc.predict_board(img, flipped=False)
print("Inferred FEN:", res["fen_placement"])
for r in range(8):
    row_strs = []
    for c in range(8):
        p = res["grid_pieces"][r][c] or "."
        conf = res["grid_confidences"][r][c]
        row_strs.append(f"{p}:{conf:.2f}")
    print(f"Row {r}: " + " ".join(row_strs))
