"""Run the local roadmap 0.1 benchmark suite."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking import main


if __name__ == "__main__":
    raise SystemExit(main())
