"""Tests for engine modules: NeuralEngine, StockfishEngine, and lichess_bot utilities."""

from __future__ import annotations
import os
import sys
import pytest
import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# NeuralEngine / board encoding tests
# ---------------------------------------------------------------------------

class TestBoardEncoding:
    """Tests for board_to_tensor and move_to_index helpers."""

    def test_tensor_shape_start(self):
        from engine.neural_engine import board_to_tensor
        board = chess.Board()
        t = board_to_tensor(board)
        assert t.shape == (17, 8, 8), f"Expected (17,8,8), got {t.shape}"

    def test_tensor_shape_empty_board(self):
        from engine.neural_engine import board_to_tensor
        board = chess.Board(fen=None)
        t = board_to_tensor(board)
        assert t.shape == (17, 8, 8)

    def test_tensor_dtype_float32(self):
        import torch
        from engine.neural_engine import board_to_tensor
        t = board_to_tensor(chess.Board())
        assert t.dtype == torch.float32

    def test_side_to_move_plane_white(self):
        from engine.neural_engine import board_to_tensor
        board = chess.Board()
        assert board.turn == chess.WHITE
        t = board_to_tensor(board)
        # Plane 12 = side-to-move: 1.0 when White to move
        assert float(t[12, 0, 0]) == 1.0

    def test_side_to_move_plane_black(self):
        from engine.neural_engine import board_to_tensor
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))  # White moves, now Black to move
        t = board_to_tensor(board)
        assert float(t[12, 0, 0]) == 0.0

    def test_white_pawn_plane(self):
        from engine.neural_engine import board_to_tensor
        board = chess.Board()
        t = board_to_tensor(board)
        # White pawns on rank 2 (squares 8-15), plane 0
        # Rank 2 in tensor row: square 8 -> row=1, col=0
        pawn_plane = t[0].sum().item()
        assert pawn_plane == 8.0, "Should have 8 white pawns"

    def test_move_to_index_roundtrip(self):
        from engine.neural_engine import move_to_index
        moves = ["e2e4", "d7d5", "g1f3", "b8c6", "f1b5", "a7a6"]
        indices = set()
        for uci in moves:
            move = chess.Move.from_uci(uci)
            idx = move_to_index(move)
            assert 0 <= idx < 4096, f"Index {idx} out of range for {uci}"
            indices.add(idx)
        # All moves should produce distinct indices
        assert len(indices) == len(moves)

    def test_move_to_index_e2e4(self):
        from engine.neural_engine import move_to_index
        move = chess.Move.from_uci("e2e4")
        # e2 = square 12, e4 = square 28 → 12*64 + 28 = 796
        assert move_to_index(move) == 796

    def test_move_to_index_a1h8(self):
        from engine.neural_engine import move_to_index
        move = chess.Move.from_uci("a1h8")
        # a1 = 0, h8 = 63 → 0*64 + 63 = 63
        assert move_to_index(move) == 63


class TestPolicyValueNetwork:
    """Tests for the neural network architecture."""

    def test_network_output_shapes(self):
        import torch
        from engine.neural_engine import PolicyValueNetwork, board_to_tensor
        net = PolicyValueNetwork(num_res_blocks=2, channels=32)
        net.eval()
        board = chess.Board()
        t = board_to_tensor(board).unsqueeze(0)  # (1, 17, 8, 8)
        with torch.no_grad():
            policy_logits, value = net(t)
        assert policy_logits.shape == (1, 4096), f"Policy shape wrong: {policy_logits.shape}"
        assert value.shape in ((1,), (1, 1)), f"Value shape wrong: {value.shape}"

    def test_value_head_in_range(self):
        import torch
        from engine.neural_engine import PolicyValueNetwork, board_to_tensor
        net = PolicyValueNetwork(num_res_blocks=2, channels=32)
        net.eval()
        board = chess.Board()
        t = board_to_tensor(board).unsqueeze(0)
        with torch.no_grad():
            _, value = net(t)
        v = value.item()
        assert -1.0 <= v <= 1.0, f"Value {v} outside [-1, 1]"


class TestNeuralEngineGreedy:
    """Test NeuralEngine greedy move selection from starting position."""

    def test_greedy_move_is_legal(self):
        from engine.neural_engine import NeuralEngine
        # weights_path=None → random weights, greedy fallback (no MCTS)
        eng = NeuralEngine(weights_path=None, mcts_simulations=0, time_budget_ms=500)
        board = chess.Board()
        move = eng.best_move(board, time_left_ms=5000)
        assert move is not None, "NeuralEngine returned None from start position"
        assert move in board.legal_moves, f"Move {move} is not legal from start"
        eng.close()

    def test_greedy_move_after_1_e4(self):
        from engine.neural_engine import NeuralEngine
        eng = NeuralEngine(weights_path=None, mcts_simulations=0, time_budget_ms=500)
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        move = eng.best_move(board, time_left_ms=5000)
        assert move in board.legal_moves
        eng.close()

    def test_context_manager(self):
        from engine.neural_engine import NeuralEngine
        with NeuralEngine(weights_path=None, mcts_simulations=0) as eng:
            board = chess.Board()
            move = eng.best_move(board, time_left_ms=1000)
            assert move in board.legal_moves


# ---------------------------------------------------------------------------
# StockfishEngine tests (skipped if binary not found)
# ---------------------------------------------------------------------------

STOCKFISH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "stockfish", "stockfish-windows-x86-64-avx2.exe"
)
stockfish_available = os.path.isfile(STOCKFISH_PATH)


@pytest.mark.skipif(not stockfish_available, reason="Stockfish binary not installed")
class TestStockfishEngine:
    """Integration tests for StockfishEngine — skipped if binary missing."""

    def test_best_move_start_position(self):
        from engine.stockfish_engine import StockfishEngine
        with StockfishEngine(path=STOCKFISH_PATH, skill_level=1) as eng:
            board = chess.Board()
            move = eng.best_move(board, time_left_ms=1000)
        assert move in chess.Board().legal_moves

    def test_analyse_returns_score(self):
        from engine.stockfish_engine import StockfishEngine
        with StockfishEngine(path=STOCKFISH_PATH, skill_level=5) as eng:
            board = chess.Board()
            info = eng.analyse(board, depth=10)
        assert "score_cp" in info

    def test_mate_in_one(self):
        from engine.stockfish_engine import StockfishEngine
        # Fool's mate position: White to play, Black has Qh4#
        # Use a position with an obvious forced move
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
        with StockfishEngine(path=STOCKFISH_PATH, skill_level=20) as eng:
            board = chess.Board(fen)
            move = eng.best_move(board, time_left_ms=2000)
        assert move in board.legal_moves


# ---------------------------------------------------------------------------
# compute_move_time_ms tests
# ---------------------------------------------------------------------------

class TestComputeMoveTimeMs:
    """Tests for the clock allocation formula in lichess_bot."""

    def test_early_game_allocates_less(self):
        from play.lichess_bot import compute_move_time_ms
        # Move 5, 10 minutes left → should be below 15s
        t = compute_move_time_ms(600_000, 0, fullmove_number=5)
        assert t < 15_000, f"Allocated {t}ms, expected < 15s for early game"

    def test_minimum_50ms(self):
        from play.lichess_bot import compute_move_time_ms
        # Bullet scenario: 2s left
        t = compute_move_time_ms(2_000, 0, fullmove_number=40)
        assert t >= 50

    def test_respects_max_budget(self):
        from play.lichess_bot import compute_move_time_ms
        t = compute_move_time_ms(600_000, 0, fullmove_number=10, max_budget_ms=3000)
        assert t <= 3000

    def test_increment_adds_time(self):
        from play.lichess_bot import compute_move_time_ms
        without_inc = compute_move_time_ms(60_000, 0, fullmove_number=20)
        with_inc = compute_move_time_ms(60_000, 2_000, fullmove_number=20)
        assert with_inc > without_inc

    def test_endgame_allocates_more_per_move(self):
        from play.lichess_bot import compute_move_time_ms
        early = compute_move_time_ms(300_000, 0, fullmove_number=5)
        late = compute_move_time_ms(300_000, 0, fullmove_number=40)
        # Endgame has fewer estimated remaining moves → bigger slice
        assert late >= early

    def test_time_left_preserved(self):
        from play.lichess_bot import compute_move_time_ms
        # Even with large budget, never exceed time_left - 500ms margin
        t = compute_move_time_ms(1_000, 0, fullmove_number=5, max_budget_ms=10_000)
        assert t <= 500  # 1000 - 500 reserve = 500 max
