"""
Local test mode: screen-reads Chess.com board, detects position, generates FEN,
displays live HUD, and optionally auto-plays moves via pyautogui.

Uses TemplateClassifier (glass sprites on ocean board) as primary recognizer,
with CNN as fallback for uncertain squares.
"""

from __future__ import annotations
import os, sys, time, json
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import cv2
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision.capture     import FrameCapture
from vision.board_detect import BoardDetector
from vision.template_classifier import TemplateClassifier
from vision.state       import GameStateManager
from vision.overlay     import draw_hud_overlay

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# ── Screen-coordinate helpers ─────────────────────────────────────────────────

def get_screen_coords(
    square_name: str,
    board_region: Tuple[int, int, int, int],  # (left, top, width, height) in screen px
    flipped: bool = False,
) -> Tuple[int, int]:
    """
    Returns the screen (x, y) center for a named square (e.g. 'e4').

    Args:
        square_name: Algebraic square name.
        board_region: (left, top, width, height) of the board on screen.
        flipped: True if Black is at the bottom.
    """
    from vision.squares import get_grid_coords
    row, col = get_grid_coords(square_name, flipped=flipped)
    left, top, width, height = board_region
    step_x = width  / 8.0
    step_y = height / 8.0
    x = int(left + (col + 0.5) * step_x)
    y = int(top  + (row + 0.5) * step_y)
    return x, y


def corners_to_region(
    corners: Optional[np.ndarray],
    cfg_region: Optional[list],
) -> Optional[Tuple[int, int, int, int]]:
    """Converts detected corners + optional region offset to (left, top, w, h)."""
    if corners is None:
        return None
    offset_x = int(cfg_region[0]) if cfg_region else 0
    offset_y = int(cfg_region[1]) if cfg_region else 0
    tl = corners[0]
    br = corners[2]
    left  = int(tl[0]) + offset_x
    top   = int(tl[1]) + offset_y
    width  = int(br[0] - tl[0])
    height = int(br[1] - tl[1])
    return left, top, width, height


def execute_mouse_move(
    move: Any,
    board_region: Tuple[int, int, int, int],
    flipped: bool = False,
    promotion_piece: str = "q",
) -> None:
    """
    Executes a chess move via pyautogui drag-and-drop on the browser board.

    Uses click-drag + destination click to handle both drag and click-to-move modes.
    """
    import pyautogui
    import chess

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.03

    from_sq = chess.square_name(move.from_square)
    to_sq   = chess.square_name(move.to_square)

    x1, y1 = get_screen_coords(from_sq, board_region, flipped=flipped)
    x2, y2 = get_screen_coords(to_sq,   board_region, flipped=flipped)

    print(f"    Mouse: {from_sq}({x1},{y1}) → {to_sq}({x2},{y2})", flush=True)

    # Strategy: mouseDown on source → smooth drag → mouseUp on destination
    pyautogui.moveTo(x1, y1, duration=0.08)
    time.sleep(0.06)
    pyautogui.mouseDown(button="left")
    time.sleep(0.10)
    pyautogui.moveTo(x2, y2, duration=0.25)
    time.sleep(0.08)
    pyautogui.mouseUp(button="left")
    time.sleep(0.10)

    # Click destination once to confirm drop (handles click-to-move as well)
    pyautogui.click(x2, y2)

    # Promotion: click the Queen tile that pops up
    if getattr(move, "promotion", None) is not None:
        time.sleep(0.35)
        pyautogui.click(x2, y2)


# ── Live loop ─────────────────────────────────────────────────────────────────

def run_live_loop(
    config_dict: Optional[Dict[str, Any]] = None,
    auto_move: bool = False,
) -> None:
    """
    Runs the continuous vision + engine loop.

    Key bindings:
      q / ESC  — quit
      a        — toggle auto-move
      f        — flip board orientation
      r        — resync game state from current FEN
      s        — save debug snapshot
    """
    cfg         = config_dict or load_config()
    vision_cfg  = cfg.get("vision", {})
    model_cfg   = cfg.get("model", {})
    match_cfg   = cfg.get("match", {})
    engine_cfg  = cfg.get("engine", {})

    source      = vision_cfg.get("capture_source", "screen")
    cfg_region  = vision_cfg.get("screen_region", None)   # [left,top,w,h] or null
    flipped     = vision_cfg.get("flipped", False)
    user_color  = match_cfg.get("user_color", "white").lower()
    weights_path = model_cfg.get("weights_path", "models/classifier.pt")

    print("[*] Starting Chess Vision AI - Live HUD")
    print("    Keys: q=quit | m/Space=PLAY MOVE NOW | a=auto-move | f=flip | r=resync | s=snapshot")
    if auto_move:
        print("[!] AUTO-MOVE ENABLED — AI will play moves on your screen!")

    # ── Initialise modules ────────────────────────────────────────────────────
    cap        = FrameCapture(source=source, region=cfg_region)
    detector   = BoardDetector(target_size=512)
    classifier = TemplateClassifier(
        ncc_threshold=0.55,
        cnn_fallback=True,
        cnn_weights=weights_path,
    )
    game_state = GameStateManager()

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = None
    try:
        eng_type = engine_cfg.get("type", "stockfish")
        if eng_type == "stockfish":
            from engine.stockfish_engine import StockfishEngine
            engine = StockfishEngine(
                path=engine_cfg.get("stockfish_path"),
                skill_level=engine_cfg.get("skill_level", 10),
                time_budget_ms=engine_cfg.get("time_budget_ms", 1000),
            )
        else:
            from engine.neural_engine import NeuralEngine
            engine = NeuralEngine(
                weights_path=engine_cfg.get("policy_weights_path"),
                mcts_simulations=engine_cfg.get("mcts_simulations", 50),
                time_budget_ms=1000,
            )
        print(f"[*] Engine ({eng_type}) ready.", flush=True)
    except Exception as e:
        print(f"[!] Engine unavailable: {e}", flush=True)

    # ── State variables ───────────────────────────────────────────────────────
    fps             = 0.0
    frame_count     = 0
    last_fps_time   = time.time()
    last_inferred   : Optional[str] = None
    suggested_move  : Optional[str] = None

    stable_frames   = 0
    last_seen_fen   = ""
    last_moved_fen  = ""   # FEN after our last auto-move (avoid re-doing same move)
    initial_synced  = False
    board_region    : Optional[Tuple[int, int, int, int]] = None

    # Pre-compute board region from config if given
    if cfg_region is not None and len(cfg_region) == 4:
        board_region = tuple(cfg_region[:4])  # type: ignore
        print(f"[*] Using fixed board region from config: {board_region}")

    def trigger_move(reason: str = "manual"):
        nonlocal last_moved_fen, last_inferred, suggested_move, stable_frames
        if engine is None:
            print("[!] Cannot move: engine offline")
            return
        if board_region is None:
            print("[!] Cannot move: board region not detected yet on screen")
            return

        import chess
        our_turn = (chess.WHITE if user_color == "white" else chess.BLACK)
        if game_state.board.turn != our_turn:
            print(f"[!] Not our turn! (Our color: {user_color.upper()}, Turn: {'WHITE' if game_state.board.turn==chess.WHITE else 'BLACK'}). Waiting for opponent.")
            return

        try:
            best = engine.best_move(game_state.board, time_left_ms=1200)
            if best is not None and best in game_state.board.legal_moves:
                print(f"[{reason.upper()}] Playing GM move: {best.uci()} ({chess.square_name(best.from_square)} -> {chess.square_name(best.to_square)})", flush=True)
                execute_mouse_move(best, board_region, flipped=flipped)
                
                # Advance internal state
                game_state.apply_move(best)
                last_moved_fen = game_state.board.board_fen()
                last_inferred = best.uci()
                suggested_move = None
                stable_frames = 0

                # Verification: give the browser animation ~400ms to finish, then check if piece arrived
                time.sleep(0.40)
                v_frame = cap.capture_frame()
                v_warped, _ = detector.detect_and_warp(v_frame, fixed_region=cfg_region)
                if v_warped is not None:
                    v_preds = classifier.predict_board(v_warped, flipped=flipped)
                    v_fen = v_preds["fen_placement"]
                    expected_placement = game_state.board.board_fen()
                    if v_fen == expected_placement:
                        print(f"    [VERIFIED] Move {best.uci()} confirmed landed on screen! (FEN matches)", flush=True)
                    else:
                        print(f"    [RETRY] Move might have slipped. Retrying click {chess.square_name(best.to_square)}...", flush=True)
                        # Re-click destination to ensure drop registered
                        tx, ty = get_screen_coords(chess.square_name(best.to_square), board_region, flipped=flipped)
                        import pyautogui
                        pyautogui.click(tx, ty)
                        time.sleep(0.20)
            else:
                print(f"[!] Engine move not legal or None: {best} in position {game_state.board.fen()}")
        except Exception as exc:
            print(f"[!] Error playing move ({reason}): {exc}", flush=True)

    try:
        while True:
            frame = cap.capture_frame()

            # ── Board detection ───────────────────────────────────────────────
            warped, corners = detector.detect_and_warp(frame, fixed_region=cfg_region)
            if warped is None:
                # Fall back to configured region crop
                if board_region is not None:
                    l, t, w, h = board_region
                    crop = frame[t:t+h, l:l+w]
                    warped = cv2.resize(crop, (512, 512)) if crop.size > 0 else None
                if warped is None:
                    cv2.imshow("Chess Vision AI - Live HUD", frame)
                    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                        break
                    time.sleep(0.03)
                    continue

            # Update board region from detected corners for mouse clicks
            if corners is not None and board_region is None:
                board_region = corners_to_region(corners, cfg_region)
                print(f"[*] Board region auto-detected: {board_region}")

            # ── Classification ────────────────────────────────────────────────
            preds       = classifier.predict_board(warped, flipped=flipped)
            current_fen = preds["fen_placement"]

            # ── Stability tracking ────────────────────────────────────────────
            if current_fen == last_seen_fen:
                stable_frames += 1
            else:
                stable_frames = 0
                last_seen_fen = current_fen

            # ── Initial sync & Auto Color Detection ───────────────────────────
            if not initial_synced and stable_frames >= 2:
                # Bottom rank in captured view is row 7 (ranks 6 and 7)
                # Count white vs black pieces on bottom 2 rows (row 6 and 7)
                bottom_pieces = [preds["grid_pieces"][r][c] for r in (6, 7) for c in range(8) if preds["grid_pieces"][r][c]]
                w_count = sum(1 for p in bottom_pieces if p.isupper())
                b_count = sum(1 for p in bottom_pieces if p.islower())

                if b_count > w_count:
                    user_color = "black"
                    flipped = True
                else:
                    user_color = "white"
                    flipped = False

                # On Chess.com:
                # If we are White, it's White's turn initially ('w')
                # If we are Black, White moves first ('w'), so opponent moves first!
                turn_char = "w"  # Chess always starts with White's move
                try:
                    game_state.resync_from_fen(f"{current_fen} {turn_char} - - 0 1")
                    print(f"[*] Board synced: {current_fen} | Our Color: {user_color.upper()} (bottom) | Turn: {turn_char.upper()}", flush=True)
                    initial_synced = True
                    if engine is not None and user_color == "white":
                        best = engine.best_move(game_state.board, time_left_ms=1000)
                        if best:
                            suggested_move = best.uci()
                except Exception as exc:
                    print(f"[!] Sync failed: {exc}", flush=True)

            # ── Move inference ────────────────────────────────────────────────
            if initial_synced:
                move, conf = game_state.infer_move_from_detected_grid(
                    preds["grid_pieces"], flipped=flipped
                )
                # Only acknowledge opponent move when the screen is completely stable (>= 3 frames)
                # to prevent jumping mid-animation
                if move is not None and conf >= 0.70 and stable_frames >= 3:
                    last_inferred = move.uci()
                    game_state.apply_move(move)
                    turn_who = "WHITE" if game_state.board.turn == chess.WHITE else "BLACK"
                    print(f"[LIVE] Opponent completed move: {move.uci()} (conf={conf:.2f}) -> Our turn to play!", flush=True)
                    if engine is not None:
                        try:
                            best = engine.best_move(game_state.board, time_left_ms=1200)
                            if best:
                                suggested_move = best.uci()
                        except Exception:
                            pass
                elif stable_frames >= 5 and current_fen != game_state.board.board_fen():
                    # Check if board drifted or opponent move wasn't cleanly detected
                    turn_char = "w" if game_state.board.turn == chess.WHITE else "b"
                    try:
                        game_state.resync_from_fen(f"{current_fen} {turn_char} - - 0 1")
                        print(f"[*] Re-synced board to: {current_fen} (Turn: {turn_char.upper()})", flush=True)
                        if engine is not None:
                            best = engine.best_move(game_state.board, time_left_ms=1000)
                            if best:
                                suggested_move = best.uci()
                    except Exception:
                        pass

            # ── Auto-move ─────────────────────────────────────────────────────
            if engine is not None and initial_synced and stable_frames >= 4:
                import chess
                our_turn = (chess.WHITE if user_color == "white" else chess.BLACK)
                is_our_turn = (game_state.board.turn == our_turn)
                board_changed = (game_state.board.board_fen() != last_moved_fen)
                not_game_over = not game_state.board.is_game_over()

                # Extra check: wait 0.15s after opponent piece settles to make sure Chess.com is ready
                if auto_move and is_our_turn and board_changed and not_game_over and board_region is not None:
                    time.sleep(0.15)
                    trigger_move("auto")

            # ── FPS ───────────────────────────────────────────────────────────
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count  = 0
                last_fps_time = now

            # ── HUD render ────────────────────────────────────────────────────
            hud = draw_hud_overlay(
                warped, preds, fps=fps,
                last_move=last_inferred,
                suggested_move=suggested_move,
                auto_move=auto_move,
            )
            cv2.imshow("Chess Vision AI - Live HUD", hud)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), 27]:
                break
            elif key == ord("m") or key == ord(" "):  # Manual move trigger
                print("[*] 'm' key pressed -> Executing move immediately...")
                trigger_move("manual")
            elif key == ord("a"):
                auto_move = not auto_move
                print(f"[*] Auto-move: {'ON' if auto_move else 'OFF'}")
            elif key == ord("f"):
                flipped = not flipped
                user_color = "black" if flipped else "white"
                print(f"[*] Orientation & color switched: playing as {user_color.upper()} ({'Black' if flipped else 'White'} at bottom)")
                initial_synced = False   # re-sync after flip
            elif key == ord("c"):
                user_color = "black" if user_color == "white" else "white"
                print(f"[*] Our piece color toggled to: {user_color.upper()}")
            elif key == ord("r"):
                # Count bottom pieces to set correct color
                bottom_pieces = [preds["grid_pieces"][r][c] for r in (6, 7) for c in range(8) if preds["grid_pieces"][r][c]]
                w_count = sum(1 for p in bottom_pieces if p.isupper())
                b_count = sum(1 for p in bottom_pieces if p.islower())
                user_color = "black" if b_count > w_count else "white"
                flipped = (user_color == "black")

                # If starting new game: White always moves first
                turn_char = "w" if "rnbqkbnr" in current_fen.lower() else ("w" if user_color == "white" else "b")
                try:
                    game_state.resync_from_fen(f"{current_fen} {turn_char} - - 0 1")
                    print(f"[*] Manual resync: {current_fen} | Our Color: {user_color.upper()} | Turn: {turn_char.upper()}")
                    if engine is not None and ((user_color == "white" and turn_char == "w") or (user_color == "black" and turn_char == "b")):
                        best = engine.best_move(game_state.board, time_left_ms=1000)
                        if best:
                            suggested_move = best.uci()
                except Exception as exc:
                    print(f"[!] Resync error: {exc}")
            elif key == ord("s"):
                os.makedirs("debug_frames", exist_ok=True)
                snap = f"debug_frames/snap_{int(time.time())}.png"
                cv2.imwrite(snap, hud)
                print(f"[*] Snapshot: {snap}")

    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()


def run_single_frame_test(config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Captures one frame, runs classification, saves debug image."""
    cfg         = config_dict or load_config()
    vision_cfg  = cfg.get("vision", {})
    model_cfg   = cfg.get("model", {})

    source      = vision_cfg.get("capture_source", "screen")
    cfg_region  = vision_cfg.get("screen_region", None)
    flipped     = vision_cfg.get("flipped", False)
    weights_path = model_cfg.get("weights_path", "models/classifier.pt")

    with FrameCapture(source=source, region=cfg_region) as cap:
        frame = cap.capture_frame()

    detector   = BoardDetector(target_size=512)
    classifier = TemplateClassifier(cnn_fallback=True, cnn_weights=weights_path)
    warped, _  = detector.detect_and_warp(frame, fixed_region=cfg_region)
    if warped is None:
        h, w = frame.shape[:2]
        s = min(h, w)
        warped = cv2.resize(frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], (512, 512))

    preds = classifier.predict_board(warped, flipped=flipped)
    hud   = draw_hud_overlay(warped, preds, fps=0.0)

    os.makedirs("debug_frames", exist_ok=True)
    cv2.imwrite("debug_frames/single_frame_result.png", hud)
    print(f"FEN: {preds['fen_placement']}")
    print("Saved: debug_frames/single_frame_result.png")

    cv2.imshow("Chess Vision AI - Single Frame", hud)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return preds


if __name__ == "__main__":
    run_live_loop()
