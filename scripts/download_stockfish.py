"""Download and setup the official Stockfish binary for Windows."""

import os
import io
import zipfile
import urllib.request

STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/"
    "stockfish-windows-x86-64-universal.zip"
)
DEST_DIR = "stockfish"
EXE_NAME = "stockfish-windows-x86-64-avx2.exe"


def download_stockfish() -> None:
    os.makedirs(DEST_DIR, exist_ok=True)
    target_path = os.path.join(DEST_DIR, EXE_NAME)

    if os.path.exists(target_path):
        print(f"[*] Stockfish binary already installed at: {target_path}")
        return

    print(f"[*] Downloading official Stockfish binary from: {STOCKFISH_URL}")
    req = urllib.request.Request(STOCKFISH_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        print(f"[*] Extracting to {target_path}...")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for member in z.namelist():
                if member.endswith(".exe"):
                    with open(target_path, "wb") as f:
                        f.write(z.read(member))
                    print(f"[+] Successfully installed Stockfish to: {target_path}")
                    return
    except Exception as e:
        print(f"[!] Error downloading Stockfish: {e}")


if __name__ == "__main__":
    download_stockfish()
