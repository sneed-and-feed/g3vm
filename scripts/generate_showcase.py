"""
scripts/generate_showcase.py: Run the G3VM 4-scenario benchmark suite, verify
Theorems 2.1-2.6 (`OcclusionTriad.lean`), and generate `results/g3vm_leapfrog_showcase.png`,
`results/g3vm_well_log_beltrami.png`, and `results/g3vm_benchmark_summary.json`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g3vm.cli import run_benchmark_suite


if __name__ == "__main__":
    run_benchmark_suite(grid_size=48, output_dir=ROOT / "results")
