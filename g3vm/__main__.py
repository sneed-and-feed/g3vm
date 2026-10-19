from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from g3vm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
