from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
