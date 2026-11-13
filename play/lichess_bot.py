"""Lichess Bot API integration for real-time chess games.

This module connects to the Lichess Bot API using a BOT account token
read from the LICHESS_BOT_TOKEN environment variable (never hardcoded).

Features:
  - Accepts configurable challenges (time control, variant, rated/casual)
  - Clock-aware dynamic time management per move
  - Draw offer and resignation handling
  - Pluggable engine: Stockfish or Neural via config.yaml
  - Full PGN logging per game with think-time and evaluation annotations
  - Graceful reconnect on network errors

Usage:
    # Set your Lichess BOT token (never hardcode it)
    export LICHESS_BOT_TOKEN=lip_yourtokenhere   # Linux/macOS
    $env:LICHESS_BOT_TOKEN = "lip_yourtokenhere" # PowerShell

    python play/lichess_bot.py
    python play/lichess_bot.py --engine stockfish --skill 15
    python play/lichess_bot.py --engine neural --mcts 200
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import argparse
import datetime
from typing import Optional, Dict, Any, List

import chess
import chess.pgn
import berserk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _load_engine(engine_type: str, config: Dict[str, Any]):
    """Instantiates the configured engine backend."""
    engine_cfg = config.get("engine", {})

    if engine_type == "stockfish":
        from engine.stockfish_engine import StockfishEngine
        return StockfishEngine(
            path=engine_cfg.get("stockfish_path"),
            skill_level=engine_cfg.get("skill_level", 10),
            uci_elo=engine_cfg.get("uci_elo"),
            time_budget_ms=engine_cfg.get("time_budget_ms", 1000),
        )

    elif engine_type == "neural":
        from engine.neural_engine import NeuralEngine
        return NeuralEngine(
            weights_path=engine_cfg.get("policy_weights_path", "models/policy_value.pt"),
            mcts_simulations=engine_cfg.get("mcts_simulations", 100),
            time_budget_ms=engine_cfg.get("time_budget_ms", 2000),
        )

    raise ValueError(f"Unknown engine type: {engine_type!r}. Use 'stockfish' or 'neural'.")


# ---------------------------------------------------------------------------
# Time management
# ---------------------------------------------------------------------------

def compute_move_time_ms(
    time_left_ms: int,
    increment_ms: int,
    fullmove_number: int,
    max_budget_ms: int = 10000,
) -> int:
    """
    Dynamic clock allocation formula.

    Uses an estimated remaining moves horizon, reduced by game phase.
    Returns move think time in milliseconds.
    """
    # Estimate remaining moves based on game phase
    if fullmove_number < 10:
        estimated_remaining = 40
    elif fullmove_number < 25:
        estimated_remaining = 30
    else:
        estimated_remaining = 20

    base_time = time_left_ms / estimated_remaining
    with_increment = base_time + increment_ms * 0.8

    # Always keep at least 500ms in reserve
    allocated = min(with_increment, time_left_ms - 500, max_budget_ms)
    return max(50, int(allocated))


# ---------------------------------------------------------------------------
# PGN Logging
# ---------------------------------------------------------------------------

class GameLogger:
    """Records game moves, think times, and evaluations as a PGN string."""

    def __init__(
        self,
        bot_username: str,
        opponent: str,
        time_control: str,
        rated: bool,
        log_dir: str = "pgn_logs",
    ) -> None:
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = "Lichess Bot Game"
        self.game.headers["Site"] = "lichess.org"
        self.game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        self.game.headers["White"] = bot_username
        self.game.headers["Black"] = opponent
        self.game.headers["TimeControl"] = time_control
        self.game.headers["Termination"] = "Normal"
        if not rated:
            self.game.headers["Event"] = "Lichess Bot Casual Game"

        self.node = self.game
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._moves: List[Dict[str, Any]] = []

    def add_move(
        self,
        move: chess.Move,
        think_ms: int = 0,
        eval_cp: Optional[int] = None,
        comment: str = "",
    ) -> None:
        """Records a move with optional think time and evaluation annotation."""
        self.node = self.node.add_variation(move)
        parts = []
        if think_ms > 0:
            parts.append(f"[%clk {think_ms / 1000:.1f}s]")
        if eval_cp is not None:
            parts.append(f"eval={eval_cp / 100:.2f}")
        if comment:
            parts.append(comment)
        if parts:
            self.node.comment = " ".join(parts)
        self._moves.append({"move": move.uci(), "think_ms": think_ms, "eval_cp": eval_cp})

    def save(self, game_id: str, result: str = "*") -> str:
        """Writes the PGN file and returns the path."""
        self.game.headers["Result"] = result
        filename = f"{self.log_dir}/game_{game_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
        with open(filename, "w") as f:
            exporter = chess.pgn.FileExporter(f)
            self.game.accept(exporter)
        logger.info(f"[GameLogger] PGN saved: {filename}")
        return filename


# ---------------------------------------------------------------------------
# Main Bot class
# ---------------------------------------------------------------------------

class LichessBot:
    """
    Full Lichess Bot API client.

    Reads token from LICHESS_BOT_TOKEN environment variable.
    """

    def __init__(
        self,
        engine_type: str = "stockfish",
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "pgn_logs",
        accept_rated: bool = True,
        accept_casual: bool = True,
        accepted_time_controls: Optional[List[str]] = None,
        max_concurrent_games: int = 1,
    ) -> None:
        """
        Args:
            engine_type: "stockfish" or "neural".
            config: Loaded config.yaml dictionary.
            log_dir: Directory to save PGN game logs.
            accept_rated: Accept rated challenges.
            accept_casual: Accept casual challenges.
            accepted_time_controls: List of accepted speed categories, e.g. ["rapid", "classical"].
            max_concurrent_games: Maximum simultaneous games (default 1 for single-core laptop).
        """
        self.token = os.environ.get("LICHESS_BOT_TOKEN", "").strip()
        if not self.token:
            raise EnvironmentError(
                "LICHESS_BOT_TOKEN environment variable is not set.\n"
                "Set it with: $env:LICHESS_BOT_TOKEN = 'lip_yourtokenhere'  (PowerShell)\n"
                "or:           export LICHESS_BOT_TOKEN=lip_yourtokenhere    (bash)"
            )

        self.config = config or {}
        self.engine_type = engine_type
        self.log_dir = log_dir
        self.accept_rated = accept_rated
        self.accept_casual = accept_casual
        self.accepted_time_controls = accepted_time_controls or ["rapid", "classical", "blitz"]
        self.max_concurrent_games = max_concurrent_games

        # Connect to Lichess API
        session = berserk.TokenSession(self.token)
        self.client = berserk.Client(session=session)

        # Verify account is a BOT account
        account = self.client.account.get()
        self.username = account["username"]
        if account.get("title") != "BOT":
            raise PermissionError(
                f"Account '{self.username}' is not a BOT account. "
                "Only BOT accounts may use the Bot API. "
                "Upgrade your account at https://lichess.org/api#tag/Bot/operation/botAccountUpgrade"
            )

        logger.info(f"[LichessBot] Connected as BOT: {self.username}")

        # Lazy engine instantiation (created per game to avoid stale state)
        self.engine = _load_engine(engine_type, self.config)
        self._active_games: Dict[str, bool] = {}

    def _should_accept(self, event: Dict[str, Any]) -> bool:
        """Determines whether to accept an incoming challenge."""
        challenge = event.get("challenge", {})
        speed = challenge.get("speed", "")
        rated = challenge.get("rated", False)
        variant = challenge.get("variant", {}).get("key", "standard")

        # Only accept standard chess
        if variant != "standard":
            logger.info(f"[Bot] Declining non-standard variant: {variant}")
            return False

        # Time control check
        if speed not in self.accepted_time_controls:
            logger.info(f"[Bot] Declining time control: {speed}")
            return False

        # Rated/casual preference
        if rated and not self.accept_rated:
            logger.info("[Bot] Declining rated game (accept_rated=False)")
            return False
        if not rated and not self.accept_casual:
            logger.info("[Bot] Declining casual game (accept_casual=False)")
            return False

        # Concurrency limit
        if len(self._active_games) >= self.max_concurrent_games:
            logger.info(f"[Bot] At concurrency limit ({self.max_concurrent_games}), declining.")
            return False

        return True

    def _play_game(self, game_id: str) -> None:
        """Handles a single game from start to finish."""
        logger.info(f"[Game {game_id}] Starting game loop.")
        self._active_games[game_id] = True

        board = chess.Board()
        game_info = self.client.games.export(game_id, as_pgn=False)
        opponent = game_info.get("players", {})

        # Determine which color the bot plays
        white_id = game_info.get("players", {}).get("white", {}).get("user", {}).get("id", "")
        bot_is_white = (white_id.lower() == self.username.lower())
        bot_color = chess.WHITE if bot_is_white else chess.BLACK

        opponent_name = (
            game_info.get("players", {}).get("black" if bot_is_white else "white", {})
            .get("user", {}).get("name", "opponent")
        )

        time_control_str = game_info.get("clock", {})
        tc_str = f"{time_control_str.get('initial', 0)//1000}+{time_control_str.get('increment', 0)//1000}"

        game_logger = GameLogger(
            bot_username=self.username,
            opponent=opponent_name,
            time_control=tc_str,
            rated=game_info.get("rated", False),
            log_dir=self.log_dir,
        )
        if not bot_is_white:
            game_logger.game.headers["White"] = opponent_name
            game_logger.game.headers["Black"] = self.username

        result = "*"

        try:
            for event in self.client.bots.stream_game_state(game_id):
                event_type = event.get("type", "")

                if event_type == "gameFull":
                    # Reconstruct board state from initial moves
                    state = event.get("state", {})
                    moves_str = state.get("moves", "")
                    board = chess.Board()
                    if moves_str:
                        for uci in moves_str.split():
                            board.push_uci(uci)

                elif event_type == "gameState":
                    state = event
                    moves_str = state.get("moves", "")
                    board = chess.Board()
                    if moves_str:
                        for uci in moves_str.split():
                            board.push_uci(uci)

                    status = state.get("status", "started")
                    if status not in ("started", "created"):
                        result = state.get("winner", "*")
                        if result == "white":
                            result = "1-0"
                        elif result == "black":
                            result = "0-1"
                        elif status in ("draw", "stalemate"):
                            result = "1/2-1/2"
                        else:
                            result = "*"
                        logger.info(f"[Game {game_id}] Game ended: {status} -> {result}")
                        break

                    # Draw offer handling
                    if state.get("bdraw") or state.get("wdraw"):
                        logger.info(f"[Game {game_id}] Draw offer received.")
                        # Accept draws in clearly equal or worse positions
                        # Simple heuristic: accept if past move 40
                        if board.fullmove_number >= 40:
                            try:
                                self.client.bots.accept_draw(game_id)
                                result = "1/2-1/2"
                                break
                            except Exception:
                                pass

                    # Only move if it is our turn
                    if board.turn != bot_color:
                        continue

                    if board.is_game_over():
                        break

                    # Compute move
                    wtime = state.get("wtime", 60000)
                    btime = state.get("btime", 60000)
                    winc = state.get("winc", 0)
                    binc = state.get("binc", 0)

                    time_left_ms = wtime if bot_is_white else btime
                    increment_ms = winc if bot_is_white else binc

                    think_ms = compute_move_time_ms(
                        time_left_ms=time_left_ms,
                        increment_ms=increment_ms,
                        fullmove_number=board.fullmove_number,
                    )

                    t0 = time.time()
                    move = self.engine.best_move(board, time_left_ms=time_left_ms)
                    actual_think_ms = int((time.time() - t0) * 1000)

                    if move is None or move not in board.legal_moves:
                        logger.warning(f"[Game {game_id}] Engine returned invalid move: {move}. Resigning.")
                        try:
                            self.client.bots.resign_game(game_id)
                        except Exception:
                            pass
                        result = "0-1" if bot_is_white else "1-0"
                        break

                    # Evaluate the resulting position
                    eval_cp = None
                    if isinstance(self.engine, __import__("engine.stockfish_engine", fromlist=["StockfishEngine"]).StockfishEngine):
                        try:
                            board.push(move)
                            analysis = self.engine.analyse(board, depth=12)
                            eval_cp = analysis.get("score_cp")
                            board.pop()
                        except Exception:
                            pass

                    game_logger.add_move(move, think_ms=actual_think_ms, eval_cp=eval_cp)

                    logger.info(
                        f"[Game {game_id}] Playing {move.uci()} "
                        f"(think={actual_think_ms}ms, time_left={time_left_ms}ms)"
                    )

                    try:
                        self.client.bots.make_move(game_id, move.uci())
                    except Exception as e:
                        logger.error(f"[Game {game_id}] Failed to send move: {e}")
                        break

                    board.push(move)

                    # Automatic resignation if heavily losing (optional)
                    if eval_cp is not None and abs(eval_cp) > 1000 and board.fullmove_number > 50:
                        if (bot_is_white and eval_cp < -1000) or (not bot_is_white and eval_cp > 1000):
                            logger.info(f"[Game {game_id}] Resigning (eval={eval_cp}cp after move {board.fullmove_number}).")
                            try:
                                self.client.bots.resign_game(game_id)
                            except Exception:
                                pass
                            result = "0-1" if bot_is_white else "1-0"
                            break

                elif event_type == "chatLine":
                    # Ignore chat
                    pass

        except Exception as e:
            logger.error(f"[Game {game_id}] Unexpected error: {e}", exc_info=True)
        finally:
            game_logger.save(game_id, result=result)
            self._active_games.pop(game_id, None)
            logger.info(f"[Game {game_id}] Game loop finished. Result: {result}")

    def run(self) -> None:
        """Main event loop: listens for challenges and plays games."""
        import threading
        logger.info(f"[LichessBot] Starting event stream for BOT: {self.username}")

        while True:
            try:
                for event in self.client.bots.stream_incoming_events():
                    event_type = event.get("type", "")

                    if event_type == "challenge":
                        challenge_id = event.get("challenge", {}).get("id", "")
                        if self._should_accept(event):
                            logger.info(f"[Bot] Accepting challenge: {challenge_id}")
                            try:
                                self.client.bots.accept_challenge(challenge_id)
                            except Exception as e:
                                logger.warning(f"[Bot] Could not accept challenge {challenge_id}: {e}")
                        else:
                            logger.info(f"[Bot] Declining challenge: {challenge_id}")
                            try:
                                self.client.bots.decline_challenge(challenge_id)
                            except Exception as e:
                                logger.warning(f"[Bot] Could not decline challenge {challenge_id}: {e}")

                    elif event_type == "gameStart":
                        game_id = event.get("game", {}).get("id", "")
                        if game_id and game_id not in self._active_games:
                            logger.info(f"[Bot] Starting game thread: {game_id}")
                            t = threading.Thread(target=self._play_game, args=(game_id,), daemon=True)
                            t.start()

                    elif event_type == "gameFinish":
                        game_id = event.get("game", {}).get("id", "")
                        self._active_games.pop(game_id, None)

            except berserk.exceptions.ResponseError as e:
                logger.error(f"[Bot] API error: {e}. Reconnecting in 10s...")
                time.sleep(10)
            except Exception as e:
                logger.error(f"[Bot] Event loop error: {e}. Reconnecting in 5s...")
                time.sleep(5)

    def close(self) -> None:
        """Cleanly shuts down the engine."""
        if self.engine:
            self.engine.close()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lichess Bot - Vision-based Chess AI")
    parser.add_argument("--engine", type=str, default="stockfish", choices=["stockfish", "neural"],
                        help="Engine backend to use")
    parser.add_argument("--skill", type=int, default=10, help="Stockfish Skill Level (0-20)")
    parser.add_argument("--elo", type=int, default=None, help="UCI_Elo target for Stockfish")
    parser.add_argument("--mcts", type=int, default=100, help="MCTS simulations for Neural engine")
    parser.add_argument("--log-dir", type=str, default="pgn_logs", help="PGN output directory")
    parser.add_argument("--no-rated", action="store_true", help="Decline rated games")
    parser.add_argument("--no-casual", action="store_true", help="Decline casual games")
    parser.add_argument(
        "--speeds", type=str, default="rapid,classical,blitz",
        help="Comma-separated accepted time controls (e.g. rapid,classical)"
    )
    args = parser.parse_args()

    import yaml
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # CLI overrides
    config.setdefault("engine", {})
    if args.elo:
        config["engine"]["uci_elo"] = args.elo
    config["engine"]["skill_level"] = args.skill
    config["engine"]["mcts_simulations"] = args.mcts

    bot = LichessBot(
        engine_type=args.engine,
        config=config,
        log_dir=args.log_dir,
        accept_rated=not args.no_rated,
        accept_casual=not args.no_casual,
        accepted_time_controls=[s.strip() for s in args.speeds.split(",")],
    )

    try:
        bot.run()
    finally:
        bot.close()
