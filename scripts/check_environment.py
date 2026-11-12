"""Check core dependencies without downloading data or running experiments."""

from importlib.metadata import version
from pathlib import Path
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Repository: {Path(__file__).resolve().parents[1]}")
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib"):
        try:
            print(f"{package}: {version(package)}")
        except Exception as exc:
            print(f"{package}: not installed ({exc})")
    print("Environment check finished. No GEO datasets or ML results were produced.")


if __name__ == "__main__":
    main()
