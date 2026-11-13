"""Interactive setup and configuration GUI for Vision-based Chess AI.

Allows selecting game time controls (10m, 30m, custom), capture mode,
board orientation, engine settings, and starting live observation loop.
"""

from __future__ import annotations
import os
import sys
import yaml
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))


class ChessVisionLauncher:
    """Tkinter interactive setup menu."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Vision-Based Chess AI - Setup & Launch")
        self.root.geometry("640x720")
        self.root.minsize(580, 680)

        self.config: Dict[str, Any] = self.load_config()
        self._build_ui()

    def load_config(self) -> Dict[str, Any]:
        """Loads configuration from YAML file."""
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_config(self) -> None:
        """Saves current GUI values back to config.yaml."""
        # Update config dictionary
        match_cfg = self.config.setdefault("match", {})
        match_cfg["time_control_preset"] = self.time_preset_var.get()
        match_cfg["initial_time_seconds"] = int(float(self.custom_mins_var.get()) * 60)
        match_cfg["increment_seconds"] = int(self.increment_var.get())

        vision_cfg = self.config.setdefault("vision", {})
        vision_cfg["capture_source"] = self.source_var.get()
        vision_cfg["flipped"] = (self.orient_var.get() == "Black at bottom")
        vision_cfg["auto_detect_orientation"] = (self.orient_var.get() == "Auto-Detect")

        engine_cfg = self.config.setdefault("engine", {})
        engine_cfg["type"] = self.engine_type_var.get()
        engine_cfg["skill_level"] = int(self.skill_var.get())
        engine_cfg["time_budget_ms"] = int(self.time_budget_var.get())

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        # Header Banner
        header_frame = tk.Frame(self.root, bg="#1e272e", pady=12)
        header_frame.pack(fill=tk.X)
        title_lbl = tk.Label(
            header_frame,
            text="CHESS VISION AI - SETUP",
            font=("Segoe UI", 16, "bold"),
            fg="#00d2d3",
            bg="#1e272e",
        )
        title_lbl.pack()
        subtitle_lbl = tk.Label(
            header_frame,
            text="Configure match parameters, screen capture, and engine prior to match start",
            font=("Segoe UI", 9),
            fg="#c8d6e5",
            bg="#1e272e",
        )
        subtitle_lbl.pack()

        # Main scrollable or padded container
        main_frame = ttk.Frame(self.root, padding="16 12 16 12")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------- Section 1: Match Time Controls ----------------
        tc_group = ttk.LabelFrame(main_frame, text=" 1. Match Time Control (Chess.com) ", padding="10")
        tc_group.pack(fill=tk.X, pady=6)

        self.time_preset_var = tk.StringVar(value="10m")
        r_10m = ttk.Radiobutton(tc_group, text="10 min Rapid (Recommended)", value="10m", variable=self.time_preset_var, command=self._on_preset_change)
        r_30m = ttk.Radiobutton(tc_group, text="30 min Classical / Rapid", value="30m", variable=self.time_preset_var, command=self._on_preset_change)
        r_custom = ttk.Radiobutton(tc_group, text="Custom Clock", value="custom", variable=self.time_preset_var, command=self._on_preset_change)

        r_10m.grid(row=0, column=0, sticky=tk.W, padx=8, pady=2)
        r_30m.grid(row=0, column=1, sticky=tk.W, padx=8, pady=2)
        r_custom.grid(row=0, column=2, sticky=tk.W, padx=8, pady=2)

        custom_box = ttk.Frame(tc_group)
        custom_box.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=6, padx=8)

        ttk.Label(custom_box, text="Minutes:").pack(side=tk.LEFT, padx=4)
        self.custom_mins_var = tk.StringVar(value="10")
        self.mins_entry = ttk.Entry(custom_box, textvariable=self.custom_mins_var, width=6)
        self.mins_entry.pack(side=tk.LEFT, padx=4)

        ttk.Label(custom_box, text="Increment (s):").pack(side=tk.LEFT, padx=8)
        self.increment_var = tk.StringVar(value="0")
        self.inc_entry = ttk.Entry(custom_box, textvariable=self.increment_var, width=6)
        self.inc_entry.pack(side=tk.LEFT, padx=4)

        # ---------------- Section 2: Vision & Screen Capture ----------------
        vis_group = ttk.LabelFrame(main_frame, text=" 2. Vision & Board Geometry ", padding="10")
        vis_group.pack(fill=tk.X, pady=6)

        ttk.Label(vis_group, text="Capture Source:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.source_var = tk.StringVar(value="screen")
        source_combo = ttk.Combobox(vis_group, textvariable=self.source_var, values=["screen", "webcam"], state="readonly", width=16)
        source_combo.grid(row=0, column=1, sticky=tk.W, padx=8, pady=4)

        ttk.Label(vis_group, text="Board Orientation:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.orient_var = tk.StringVar(value="Auto-Detect")
        orient_combo = ttk.Combobox(vis_group, textvariable=self.orient_var, values=["Auto-Detect", "White at bottom", "Black at bottom"], state="readonly", width=16)
        orient_combo.grid(row=1, column=1, sticky=tk.W, padx=8, pady=4)

        # ---------------- Section 3: Engine Selection ----------------
        eng_group = ttk.LabelFrame(main_frame, text=" 3. AI Engine Settings ", padding="10")
        eng_group.pack(fill=tk.X, pady=6)

        ttk.Label(eng_group, text="Engine Core:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.engine_type_var = tk.StringVar(value="stockfish")
        eng_combo = ttk.Combobox(eng_group, textvariable=self.engine_type_var, values=["stockfish", "neural"], state="readonly", width=16)
        eng_combo.grid(row=0, column=1, sticky=tk.W, padx=8, pady=4)

        ttk.Label(eng_group, text="Stockfish Skill (0-20):").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.skill_var = tk.StringVar(value="10")
        skill_spin = ttk.Spinbox(eng_group, from_=0, to=20, textvariable=self.skill_var, width=6)
        skill_spin.grid(row=1, column=1, sticky=tk.W, padx=8, pady=4)

        ttk.Label(eng_group, text="Think Budget (ms):").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.time_budget_var = tk.StringVar(value="1000")
        budget_spin = ttk.Spinbox(eng_group, from_=100, to=10000, increment=250, textvariable=self.time_budget_var, width=6)
        budget_spin.grid(row=2, column=1, sticky=tk.W, padx=8, pady=4)

        self.auto_move_var = tk.BooleanVar(value=True)
        chk_auto = ttk.Checkbutton(
            eng_group,
            text="Auto-Move (clicks pieces on screen against Bots)",
            variable=self.auto_move_var,
        )
        chk_auto.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=6)

        # ---------------- Action Buttons ----------------
        act_frame = ttk.Frame(main_frame, padding="10")
        act_frame.pack(fill=tk.X, pady=8)

        btn_test = tk.Button(
            act_frame,
            text="Calibrate & Test Board",
            command=self.run_single_test,
            bg="#2ed573",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
        )
        btn_test.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        btn_live = tk.Button(
            act_frame,
            text="Launch Live Vision HUD",
            command=self.run_live_loop,
            bg="#1e90ff",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
        )
        btn_live.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        btn_lichess = tk.Button(
            act_frame,
            text="Launch Lichess Bot",
            command=self.run_lichess_bot,
            bg="#9b59b6",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=6,
        )
        btn_lichess.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        # Footer notes
        footer_lbl = tk.Label(
            main_frame,
            text="* Note: Automated play against human players on Chess.com is strictly forbidden.\nUse this pipeline for analysis, learning, vs-computer games, and Lichess Bot API.",
            font=("Segoe UI", 8, "italic"),
            fg="#747d8c",
            justify=tk.CENTER,
        )
        footer_lbl.pack(side=tk.BOTTOM, pady=4)

    def _on_preset_change(self) -> None:
        preset = self.time_preset_var.get()
        if preset == "10m":
            self.custom_mins_var.set("10")
            self.increment_var.set("0")
        elif preset == "30m":
            self.custom_mins_var.set("30")
            self.increment_var.set("0")

    def run_single_test(self) -> None:
        """Saves config and launches single-frame board detection and classification test."""
        self.save_config()
        self.root.destroy()
        import play.local_test_mode as ltm
        ltm.run_single_frame_test()

    def run_live_loop(self) -> None:
        """Saves config and launches continuous live vision HUD."""
        self.save_config()
        auto_move = self.auto_move_var.get()
        self.root.destroy()
        import play.local_test_mode as ltm
        ltm.run_live_loop(auto_move=auto_move)

    def run_lichess_bot(self) -> None:
        """Saves config and launches the Lichess Bot API loop."""
        self.save_config()
        token = os.environ.get("LICHESS_BOT_TOKEN")
        if not token:
            messagebox.showwarning(
                "Lichess Token Required",
                "LICHESS_BOT_TOKEN environment variable is not set.\n\n"
                "Please set it before starting the bot:\n"
                "PowerShell: $env:LICHESS_BOT_TOKEN='your_token'\n"
                "CMD: set LICHESS_BOT_TOKEN=your_token\n\n"
                "You can get a bot token from https://lichess.org/account/oauth/token",
            )
            return

        self.root.destroy()
        from play.lichess_bot import LichessBot
        bot = LichessBot(engine_type=self.engine_type_var.get(), config=self.config)
        bot.run()


def launch_gui() -> None:
    root = tk.Tk()
    app = ChessVisionLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
