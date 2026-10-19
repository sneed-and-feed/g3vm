"""
g3vm.cli: Cross-platform command-line interface for G3VM
(`serve`, `benchmark`, `export`, `bundle-html`, `ingest-las`, `ingest-segy`).

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
import torch

from .engine import BeltramiWellLogShell
from .io import (
    read_las2_file,
    read_segy_cube,
    write_dev_survey,
    write_las2_file,
    write_segy_cube,
)
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
        "engine": "G3VM (DarkSnail-Geo3D) v0.2.0",
        "grid_shape": list(shape),
        "scenarios": {},
    }

    print("=" * 108)
    print(f"G3VM (DarkSnail-Geo3D) v0.2.0 3D Velocity & TTI Elastic Benchmark | Grid: {grid_size}x{grid_size}x{grid_size}")
    print("=" * 108)
    header = (
        f"{'Scenario':<20} | {'G3VM RMSE':>10} | {'RBF RMSE':>10} | {'Delta%':>8} | "
        f"{'TTI RMSE':>9} | {'Sharpness':>10} | {'DevSpike':>9} | {'<v,d_ell>':>10}"
    )
    print(header)
    print("-" * 108)

    for scen_name in SCENARIO_Descriptions:
        res = evaluate_scenario_comparison(scen_name, grid_shape=shape)
        m = res["metrics"]
        summary["scenarios"][scen_name] = m
        print(
            f"{scen_name:<20} | "
            f"{m['g3vm_vp_rmse_kms']:>10.4f} | "
            f"{m['naive_vp_rmse_kms']:>10.4f} | "
            f"-{m['rmse_reduction_pct']:>6.2f}% | "
            f"{m['g3vm_vp_tti_rmse_kms']:>9.4f} | "
            f"{m['g3vm_fault_sharpness_ratio']*100:>8.1f}% | "
            f"{m['deviated_well_spike_recovery']*100:>7.1f}% | "
            f"{m['max_orthogonality_residual']:>10.2e}"
        )

    print("=" * 108)
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
        description="G3VM (DarkSnail-Geo3D): 3D Beltrami-Triad Geological Velocity & TTI Elastic Modeler",
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

    p_export = sub.add_parser(
        "export",
        help="Export 3D velocity/TTI volume (.npz or .segy) and deviated wells (.las / .dev)",
    )
    p_export.add_argument("--scenario", default="subsalt_graben", help="Scenario name")
    p_export.add_argument("--grid-size", type=int, default=48, help="Cube resolution (default: 48)")
    p_export.add_argument(
        "--output",
        default="results/g3vm_cube.npz",
        help="Output path (.npz, .segy, .sgy, .las, or .dev)",
    )

    p_las = sub.add_parser("ingest-las", help="Ingest a CWLS LAS 2.0 well log & run Beltrami spike shielding")
    p_las.add_argument("las_file", help="Path to input .las file")
    p_las.add_argument("--output-las", default="", help="Optional output shielded .las path")

    p_segy = sub.add_parser("ingest-segy", help="Ingest a 3D SEG-Y Rev 1 velocity/seismic cube & print diagnostics")
    p_segy.add_argument("segy_file", help="Path to input .segy / .sgy file")

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
        suffix = out_path.suffix.lower()
        g3vm_out = bundle["g3vm_output"]
        scen = bundle["scenario"]

        if suffix in (".segy", ".sgy"):
            write_segy_cube(out_path, g3vm_out["vp_tti_cube_kms"])
            print(f"[+] Exported SEG-Y Rev 1 3D TTI velocity cube ({n}x{n}x{n}) to {out_path}")
            return 0
        if suffix == ".las":
            dev_well = scen["deviated_wells"][0]
            b = dev_well["beltrami"]
            write_las2_file(
                out_path,
                trajectory=dev_well["trajectory"],
                vp_kms=b.reconstructed_vp.cpu().numpy(),
                beltrami_flux=b.shielded_salience.cpu().numpy(),
            )
            print(f"[+] Exported CWLS LAS 2.0 deviated well log to {out_path}")
            return 0
        if suffix == ".dev":
            dev_well = scen["deviated_wells"][0]
            write_dev_survey(out_path, dev_well["trajectory"])
            print(f"[+] Exported directional survey (.dev) to {out_path}")
            return 0

        np.savez_compressed(
            out_path,
            g3vm_vp_kms=g3vm_out["vp_cube_kms"].cpu().numpy(),
            g3vm_vp_tti_kms=g3vm_out["vp_tti_cube_kms"].cpu().numpy(),
            g3vm_vs_kms=g3vm_out["vs_cube_kms"].cpu().numpy(),
            g3vm_rho_gcc=g3vm_out["rho_cube_gcc"].cpu().numpy(),
            g3vm_epsilon=g3vm_out["epsilon_cube"].cpu().numpy(),
            g3vm_delta=g3vm_out["delta_cube"].cpu().numpy(),
            naive_vp_kms=bundle["naive_output"]["vp_cube_kms"].cpu().numpy(),
            true_vp_kms=scen["true_vp_cube"].cpu().numpy(),
            g3vm_lithology=g3vm_out["lithology_cube"].cpu().numpy(),
            true_lithology=scen["true_litho_cube"].cpu().numpy(),
            precision_pi=scen["precision_3d"].cpu().numpy(),
        )
        print(f"[+] Exported 3D velocity & TTI elastic volume ({n}x{n}x{n}) to {out_path}")
        return 0

    if cmd == "ingest-las":
        rec = read_las2_file(args.las_file)
        vp_trace = torch.as_tensor(rec.curves["VP"], dtype=torch.float32)
        shell = BeltramiWellLogShell()
        res = shell(vp_trace)
        print(
            f"[+] Ingested LAS 2.0 well '{rec.well_name}' ({rec.well_type}, {vp_trace.numel()} samples) | "
            f"Max inclination: {rec.trajectory.inc_deg.max():.1f} deg | "
            f"Beltrami spike recovery: {res.beltrami_spike_recovery * 100:.1f}%"
        )
        if args.output_las:
            write_las2_file(
                args.output_las,
                trajectory=rec.trajectory,
                vp_kms=res.reconstructed_vp.cpu().numpy(),
                vs_kms=rec.curves.get("VS"),
                rho_gcc=rec.curves.get("RHOB"),
                epsilon=rec.curves.get("EPSILON"),
                delta=rec.curves.get("DELTA"),
                beltrami_flux=res.shielded_salience.cpu().numpy(),
            )
            print(f"[+] Saved Beltrami-shielded LAS 2.0 file to {args.output_las}")
        return 0

    if cmd == "ingest-segy":
        segy = read_segy_cube(args.segy_file)
        cube = segy["cube_zyx"]
        print(
            f"[+] Ingested SEG-Y Rev 1 3D volume: shape={cube.shape}, "
            f"format={segy['format_code']}, dt={segy['sample_interval_us']} us, "
            f"Vp range=[{cube.min():.3f}, {cube.max():.3f}] km/s"
        )
        return 0

    run_benchmark_suite(grid_size=getattr(args, "grid_size", 48), output_dir=getattr(args, "output_dir", "results"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
