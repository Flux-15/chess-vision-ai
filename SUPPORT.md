# Support & Community

Thank you for exploring **Vision-Based Chess AI**! If you need assistance, wish to discuss mathematical/vision algorithms, or have questions regarding development, please use the resources below.

---

## 💬 Community Channels

- **GitHub Discussions:** Join discussions on architecture design, MCTS optimizations, and custom board datasets:
  👉 [GitHub Discussions](https://github.com/Flux-15/chess-vision-ai/discussions)
- **GitHub Issues:** For bug reports and verified failure cases:
  👉 [Submit an Issue](https://github.com/Flux-15/chess-vision-ai/issues)

---

## ❓ Frequently Asked Questions (FAQ)

### 1. Why does board detection fail on my monitor?
- Ensure browser zoom is set to exactly **100%** (`Ctrl + 0`).
- Ensure no overlay windows, taskbars, or browser extensions obstruct the 4 outer corners of the board.
- If using multi-monitor setups, pass the specific monitor or screen region explicitly in `config.yaml`:
  ```yaml
  vision:
    screen_region: [left, top, width, height]
  ```

### 2. Can I use custom piece sets?
- Yes. The synthetic training pipeline in `data/generate_dataset.py` can load any transparent PNG sprites placed under `assets/pieces/<style_name>/`.
- Retrain the classifier with:
  ```bash
  python main.py --train --train-samples 100000 --epochs 8
  ```

### 3. How do I adjust engine strength?
- Open `config.yaml` and adjust under `engine`:
  ```yaml
  engine:
    skill_level: 10   # 0 (Novice) to 20 (Grandmaster)
    uci_elo: 1800     # Target Elo rating (overrides skill_level if set)
    time_budget_ms: 1000  # Search time per move in ms
  ```

---

## 🛡️ Fair Play & Ethical Notice
Please note that the maintainers do not provide support for bypassing commercial platform fair-play or anti-cheat detection systems. Automated testing should only be conducted on authorized developer platforms (such as the official Lichess Bot API) or against offline local computer engines.
