"""Tests for game state tracking, move inference from consecutive frames, and move legality validation."""

from __future__ import annotations
import chess
import pytest
from vision.state import GameStateManager, fen_placement_to_grid


def test_initial_state_manager() -> None:
    """Verifies GameStateManager starts at initial chess position."""
    mgr = GameStateManager()
    assert mgr.get_piece_placement() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert mgr.is_game_over() is False


def test_infer_standard_pawn_move() -> None:
    """Tests move inference for 1. e4 from starting position."""
    mgr = GameStateManager()

    # Position after 1. e4
    board_after = chess.Board()
    board_after.push_san("e4")
    grid_after = fen_placement_to_grid(board_after.board_fen())

    move, confidence = mgr.infer_move_from_detected_grid(grid_after, flipped=False)
    assert move is not None
    assert move == chess.Move.from_uci("e2e4")
    assert confidence >= 0.9

    # Apply move and check updated state
    applied = mgr.apply_move(move)
    assert applied is True
    assert mgr.get_piece_placement() == board_after.board_fen()


def test_infer_castling_move() -> None:
    """Tests inference of White kingside castling (O-O: e1g1)."""
    # Set up board where White can castle kingside
    fen_before = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5"
    board_before = chess.Board(fen_before)
    mgr = GameStateManager(initial_fen=fen_before)

    # White castles kingside: e1g1, rook jumps from h1 to f1
    board_after = board_before.copy()
    board_after.push_san("O-O")
    grid_after = fen_placement_to_grid(board_after.board_fen())

    move, confidence = mgr.infer_move_from_detected_grid(grid_after, flipped=False)
    assert move is not None
    assert move == chess.Move.from_uci("e1g1")
    assert confidence >= 0.9


def test_reject_impossible_move() -> None:
    """Tests that an impossible/illegal transition is rejected with None."""
    mgr = GameStateManager()

    # Illegal transition: White rook suddenly appears on e5 on move 1
    board_corrupt = chess.Board()
    grid_corrupt = fen_placement_to_grid(board_corrupt.board_fen())
    grid_corrupt[3][4] = "R"  # e5 occupied by white rook without path

    move, confidence = mgr.infer_move_from_detected_grid(grid_corrupt, flipped=False)
    # Cannot match any legal move cleanly
    assert move is None or confidence < 0.8


def test_resync() -> None:
    """Tests manual or automatic state resync to an arbitrary legal FEN."""
    mgr = GameStateManager()
    custom_fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    success = mgr.resync(custom_fen)
    assert success is True
    assert mgr.board.fen() == custom_fen
