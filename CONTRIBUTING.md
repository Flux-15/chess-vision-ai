# Contributing to Vision-Based Chess AI

Thank you for your interest in contributing to **Vision-Based Chess AI**! We welcome contributions ranging from computer vision algorithms, neural engine improvements, bug fixes, documentation enhancements, to testing infrastructure.

Please take a moment to review this document before submitting contributions.

---

## ⚠️ Code of Conduct & Ethical Compliance

All contributors and users are expected to adhere strictly to ethical guidelines:
1. **Academic & Fair Play Integrity:** Under no circumstances should contributions facilitate cheating or automated assistance against human players on commercial chess servers (Chess.com, Lichess, etc.).
2. **Authorized Testing Only:** All bot testing scripts must target offline engines, simulated bots, or authorized developer API endpoints (such as the official [Lichess Bot API](https://lichess.org/api#tag/Bot)).
3. **Professionalism:** Treat all participants with respect and courtesy. Harassment or discriminatory behavior will not be tolerated.

---

## How Can I Contribute?

### 1. Reporting Bugs
- Check the [GitHub Issues](https://github.com/Flux-15/chess-vision-ai/issues) tab first to see if the problem has already been reported.
- If creating a new issue, include:
  - Operating system & Python version
  - GPU model and CUDA driver version (if applicable)
  - Detailed steps to reproduce the issue
  - Complete error traceback and log output
  - Sample images or screen captures (if related to vision detection failures)

### 2. Suggesting Features & Enhancements
- Open a feature request issue describing the proposed functionality.
- Detail why the enhancement would be valuable and provide architectural considerations or math/algorithm references where applicable.

### 3. Pull Request Guidelines

1. **Fork & Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Code Style & Formatting:**
   - Adhere to **PEP 8** standards.
   - Use strict type annotations (`typing` module) for all public functions and classes.
   - Include descriptive docstrings detailing arguments, returns, and edge cases.

3. **Testing:**
   - Write unit tests for new functionality under the `tests/` directory.
   - Ensure all existing tests pass before submitting:
     ```bash
     pytest tests/ -v
     ```

4. **Commit Messages:**
   - Write clear, concise commit messages following standard conventions:
     - `feat: add HSV-based piece pre-filtering for low-contrast boards`
     - `fix: prevent race condition during opponent move animation`
     - `docs: expand mathematical derivations for homography warp`

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/chess-vision-ai.git
cd chess-vision-ai

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy
```

---

## Submitting Pull Requests

1. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a Pull Request against the `main` branch of `Flux-15/chess-vision-ai`.
3. Provide a clear description of the problem solved, architectural changes made, and test coverage verified.
