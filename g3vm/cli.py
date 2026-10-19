"""
g3vm.cli: Cross-platform command-line interface for G3VM
(`serve`, `benchmark`, `export`, `bundle-html`).

When executed as a double-clicked PyInstaller binary with no arguments on
Windows, macOS, or Linux, automatically starts the local G3VM 3D Studio server
and opens the user's default web browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
from typing import Sequence
import webbrowser

import numpy as np

from .scenarios import SCENARIO_Descriptions, evaluate_scenario_comparison
from .viewer import (
    create_studio_app,
    export_portable_html_studio,
    render_leapfrog_showcase_figure,
    render_well_log_beltrami_figure,
)


def run_benchmark_suite(
    grid_size: int = 48,
    output_dir: str | Path = "results",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (grid_size, grid_size, grid_size)

    summary = {
        "engine": "G3VM (DarkSnail-Geo3D) v0.1.0",
        "grid_shape": list(shape),
        "scenarios": {},
    }

    print("=" * 94)
    print(f"G3VM (DarkSnail-Geo3D) 3D Velocity Benchmark Suite | Grid: {grid_size}x{grid_size}x{grid_size}")
    print("=" * 94)
    header = (
        f"{'Scenario':<20} | {'G3VM RMSE':>10} | {'RBF RMSE':>10} | {'Delta%':>8} | "
        f"{'Sharpness':>10} | {'SpikeRec':>9} | {'<v,d_ell>':>10}"
    )
    print(header)
    print("-" * 94)

    for scen_name in SCENARIO_Descriptions:
        res = evaluate_scenario_comparison(scen_name, grid_shape=shape)
        m = res["metrics"]
        summary["scenarios"][scen_name] = m
        print(
            f"{scen_name:<20} | "
            f"{m['g3vm_vp_rmse_kms']:>10.4f} | "
            f"{m['naive_vp_rmse_kms']:>10.4f} | "
            f"-{m['rmse_reduction_pct']:>6.2f}% | "
            f"{m['g3vm_fault_sharpness_ratio']*100:>8.1f}% | "
            f"{m['beltrami_spike_recovery']*100:>8.1f}% | "
            f"{m['max_orthogonality_residual']:>10.2e}"
        )

    print("=" * 94)
    showcase_path = out_dir / "g3vm_leapfrog_showcase.png"
    well_path = out_dir / "g3vm_well_log_beltrami.png"
    html_path = out_dir / "g3vm_portable_studio.html"
    json_path = out_dir / "g3vm_benchmark_summary.json"

    render_leapfrog_showcase_figure(showcase_path, grid_shape=shape)
    render_well_log_beltrami_figure(well_path, grid_shape=shape)
    export_portable_html_studio(html_path, grid_shape=(24, 24, 24))

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Saved showcase figure : {showcase_path}")
    print(f"[+] Saved borehole figure : {well_path}")
    print(f"[+] Saved portable studio : {html_path}")
    print(f"[+] Saved JSON telemetry  : {json_path}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="g3vm",
        description="G3VM (DarkSnail-Geo3D): 3D Beltrami-Triad Geological Velocity Modeler",
    )
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Launch the interactive 3D G3VM Web Studio")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8055, help="Bind port (default: 8055)")
    p_serve.add_argument("--grid-size", type=int, default=32, help="Default 3D cube size (default: 32)")
    p_serve.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the default web browser on startup",
    )

    p_bench = sub.add_parser("benchmark", help="Run the 4-scenario 3D velocity benchmark & generate figures")
    p_bench.add_argument("--grid-size", type=int, default=48, help="Cube resolution (default: 48)")
    p_bench.add_argument("--output-dir", default="results", help="Output directory (default: results)")

    p_bundle = sub.add_parser(
        "bundle-html",
        help="Export a zero-install standalone HTML5 3D Studio file (runs offline in any browser)",
    )
    p_bundle.add_argument(
        "--output",
        default="results/g3vm_portable_studio.html",
        help="Output HTML file path",
    )
    p_bundle.add_argument("--grid-size", type=int, default=24, help="Embedded cube size (default: 24)")

    p_export = sub.add_parser("export", help="Export a 3D velocity cube and lithology volume to .npz")
    p_export.add_argument("--scenario", default="subsalt_graben", help="Scenario name")
    p_export.add_argument("--grid-size", type=int, default=48, help="Cube resolution (default: 48)")
    p_export.add_argument("--output", default="results/g3vm_cube.npz", help="Output .npz path")

    raw_args = list(argv) if argv is not None else sys.argv[1:]
    # If launched as a double-clicked frozen binary with no CLI args, default to `serve --open-browser`
    if not raw_args and getattr(sys, "frozen", False):
        raw_args = ["serve", "--open-browser"]

    args = parser.parse_args(raw_args)
    cmd = args.command or "benchmark"

    if cmd == "serve":
        n = max(16, min(64, int(args.grid_size)))
        app = create_studio_app(default_grid_shape=(n, n, n))
        url = f"http://{args.host}:{args.port}"
        print(f"[*] Launching G3VM 3D Subsurface Velocity Studio at {url}")
        if getattr(args, "open_browser", False):
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        app.run(host=args.host, port=args.port, debug=False)
        return 0

    if cmd == "bundle-html":
        n = max(16, min(48, int(args.grid_size)))
        out = export_portable_html_studio(args.output, grid_shape=(n, n, n))
        print(f"[+] Exported standalone zero-install HTML5 studio to {out}")
        return 0

    if cmd == "export":
        n = max(16, min(128, int(args.grid_size)))
        bundle = evaluate_scenario_comparison(args.scenario, grid_shape=(n, n, n))
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            g3vm_vp_kms=bundle["g3vm_output"]["vp_cube_kms"].cpu().numpy(),
            naive_vp_kms=bundle["naive_output"]["vp_cube_kms"].cpu().numpy(),
            true_vp_kms=bundle["scenario"]["true_vp_cube"].cpu().numpy(),
            g3vm_lithology=bundle["g3vm_output"]["lithology_cube"].cpu().numpy(),
            true_lithology=bundle["scenario"]["true_litho_cube"].cpu().numpy(),
            precision_pi=bundle["scenario"]["precision_3d"].cpu().numpy(),
        )
        print(f"[+] Exported 3D velocity volume ({n}x{n}x{n}) to {out_path}")
        return 0

    run_benchmark_suite(grid_size=getattr(args, "grid_size", 48), output_dir=getattr(args, "output_dir", "results"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
