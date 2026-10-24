#!/usr/bin/env python3
"""
scripts/evaluate_g3vm.py — Full 7-Scenario Benchmark & Verification Suite for G3VM v0.7.0.

Evaluates Scenarios 01-07:
  01: subsalt_graben          (Marmousi2 / Gulf of Mexico Subsalt Graben & Louann Diapir)
  02: thrust_anticline        (SEG/EAGE 3D Overthrust Duplex & Sub-Thrust Reef)
  03: unconformity_trap       (4D CCS / H2 Sub-Unconformity Fault Trap & Salt Cavern)
  04: egs_tbm_hazard          (Superhot EGS Granite & TBM Karst Look-Ahead)
  05: utah_forge_egs_doublet  (US DOE Utah FORGE 16A(78)-32 & 16B(78)-32 EGS Doublet)
  06: lunar_lava_tube_isru    (Lunar Lava Tube, Regolith Megascattering & Polar Ice ISRU)
  07: global_spherical_let_earth (Global Spherical Earth PREM + 3D Slab & Plume LET)

Generates:
  - results/g3vm_evaluation_summary.json
  - results/g3vm_benchmark_summary.json
  - results/g3vm_referee_audit_summary.json
  - results/g3vm_v070_let_benchmark_summary.json
  - studio/g3vm_portable_studio.html
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g3vm import (
    SCENARIO_Descriptions,
    evaluate_scenario_comparison,
    export_portable_html_studio,
    relocate_lunar_isru_moonquakes,
    relocate_utah_forge_microseismic,
    render_leapfrog_showcase_figure,
    render_well_log_beltrami_figure,
    run_global_spherical_let_benchmark,
    run_referee_audit_suite,
)


def main() -> int:
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "engine": "G3VM (DarkSnail-Geo3D) v0.7.0",
        "grid_shape": [48, 48, 48],
        "lunar_grid_shape": [32, 24, 24],
        "spherical_grid_shape": [32, 32, 32],
        "scenarios": {},
    }

    print("=" * 120)
    print(
        "G3VM (DarkSnail-Geo3D) v0.7.0 — 7-Scenario 3D Velocity, TTI Elastic, Lunar Radar & Spherical LET Evaluation"
    )
    print("=" * 120)
    header = (
        f"{'Scenario':<24} | {'Grid':<10} | {'G3VM Vp':>9} | {'RBF Vp':>9} | {'Vp Red%':>8} | "
        f"{'Sharpness':>10} | {'SpikeRec':>9} | {'<v,d_ell>':>10}"
    )
    print(header)
    print("-" * 120)

    peak_ram_lunar_mb = 0.0

    for scen_name in SCENARIO_Descriptions:
        if scen_name == "lunar_lava_tube_isru":
            grid_shape = (32, 24, 24)
        elif scen_name == "global_spherical_let_earth":
            grid_shape = (32, 32, 32)
        else:
            grid_shape = (48, 48, 48)
        is_lunar = scen_name == "lunar_lava_tube_isru"

        gc.collect()
        tracemalloc.start()
        res = evaluate_scenario_comparison(scen_name, grid_shape=grid_shape)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = float(peak_bytes) / (1024.0 * 1024.0)
        if is_lunar:
            peak_ram_lunar_mb = peak_mb

        m = dict(res["metrics"])
        m["peak_heap_ram_mb"] = round(peak_mb, 3)
        summary["scenarios"][scen_name] = m

        grid_str = f"{grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]}"
        print(
            f"{scen_name:<24} | {grid_str:<10} | "
            f"{m['g3vm_vp_rmse_kms']:>9.4f} | "
            f"{m['naive_vp_rmse_kms']:>9.4f} | "
            f"-{m['rmse_reduction_pct']:>6.2f}% | "
            f"{m['g3vm_fault_sharpness_ratio'] * 100:>8.1f}% | "
            f"{m['deviated_well_spike_recovery'] * 100:>7.1f}% | "
            f"{m['max_orthogonality_residual']:>10.2e}"
        )

    print("-" * 120)
    lm = summary["scenarios"]["lunar_lava_tube_isru"]
    print(
        f"[Lunar ISRU Telemetry] Dielectric eps_r RMSE: G3VM={lm['g3vm_eps_r_rmse']:.5f} vs Naive RBF={lm['naive_eps_r_rmse']:.5f} "
        f"(-{lm['eps_r_rmse_reduction_pct']:.2f}%) | Radar v_EM RMSE: G3VM={lm['g3vm_v_em_rmse_m_ns']:.5f} vs RBF={lm['naive_v_em_rmse_m_ns']:.5f} m/ns"
    )
    print(
        f"[Lunar ISRU Telemetry] Cross-Arch Sharpness: G3VM={lm['g3vm_arch_sharpness_ratio']*100:.1f}% vs RBF={lm['naive_arch_sharpness_ratio']*100:.1f}% | "
        f"Beltrami Spike Rec: {lm['beltrami_spike_recovery']*100:.1f}% (Naive: {lm['naive_spike_recovery']*100:.1f}%) | "
        f"Peak Heap RAM: {peak_ram_lunar_mb:.2f} MB (< 35.0 MB limit)"
    )
    print("=" * 120)

    # Strict v0.4.0 verification gates
    assert lm["rmse_reduction_pct"] >= 90.0, f"Expected >=90% Vp RMSE reduction, got {lm['rmse_reduction_pct']}%"
    assert lm["eps_r_rmse_reduction_pct"] >= 90.0, f"Expected >=90% eps_r RMSE reduction, got {lm['eps_r_rmse_reduction_pct']}%"
    assert lm["g3vm_arch_sharpness_ratio"] >= 0.98, f"Expected >=98% arch sharpness, got {lm['g3vm_arch_sharpness_ratio']}"
    assert lm["naive_arch_sharpness_ratio"] < 0.20, f"Expected <20% naive arch sharpness, got {lm['naive_arch_sharpness_ratio']}"
    assert lm["beltrami_spike_recovery"] == 1.0, f"Expected 100% Beltrami spike recovery, got {lm['beltrami_spike_recovery']}"
    assert lm["deviated_well_spike_recovery"] == 1.0, f"Expected 100% deviated spike recovery, got {lm['deviated_well_spike_recovery']}"
    assert lm["naive_spike_recovery"] < 0.20, f"Expected <20% naive spike recovery, got {lm['naive_spike_recovery']}"
    assert lm["continuity_cycle_score"] > 0.0, "Expected positive diurnal cycle score C(c) > 0"
    assert lm["nope_support_recovery"] == 1.0, f"Expected 100% H_nope support recovery, got {lm['nope_support_recovery']}"
    assert peak_ram_lunar_mb < 35.0, f"Expected peak heap RAM < 35 MB, got {peak_ram_lunar_mb:.2f} MB"

    eval_json_path = out_dir / "g3vm_evaluation_summary.json"
    bench_json_path = out_dir / "g3vm_benchmark_summary.json"
    audit_json_path = out_dir / "g3vm_referee_audit_summary.json"
    html_path = ROOT / "studio" / "g3vm_portable_studio.html"

    # Execute Mode B: Referee-Hardened Blind Holdout, Tuned Kriging & Stress-Test Suite
    audit_res = run_referee_audit_suite(grid_shape=(32, 24, 24))
    summary["referee_audit_mode_b"] = audit_res

    print("\n" + "=" * 120)
    print(
        "MODE B — BLIND LEAVE-ONE-WELL-OUT (LOWO) & UNSEEN DEVIATED WELL BENCHMARK (@ SNR = 20 dB, ZERO TARGET LEAKAGE)"
    )
    print("=" * 120)
    print(
        f"{'Scenario':<24} | {'LOWO Box':>9} | {'LOWO Krig':>10} | {'LOWO FKrig':>11} | {'LOWO G3VM':>10} | "
        f"{'Dev Krig':>9} | {'Dev G3VM':>9} | {'EdgeFid G3VM':>12}"
    )
    print("-" * 120)
    for scen_name, b_row in audit_res["blind_scenarios_snr20db"].items():
        lw = b_row["lowo_heldout_well_rmse_kms"]
        dw = b_row["heldout_deviated_well_rmse_kms"]
        ef = b_row["bounded_edge_gradient_fidelity"]
        print(
            f"{scen_name:<24} | "
            f"{lw['naive_box_smooth']:>9.4f} | "
            f"{lw['tuned_anisotropic_kriging']:>10.4f} | "
            f"{lw['fault_partitioned_kriging']:>11.4f} | "
            f"{lw['blind_darksnail_g3vm']:>10.4f} | "
            f"{dw['tuned_anisotropic_kriging']:>9.4f} | "
            f"{dw['blind_darksnail_g3vm']:>9.4f} | "
            f"{ef['blind_darksnail_g3vm'] * 100:>11.1f}%"
        )
    print("-" * 120)
    print(
        f"[Mode B Peak Heap RAM] {audit_res['peak_heap_mb']:.2f} MB (< 35.0 MB limit)"
    )
    print("\n" + "=" * 120)
    print("DOUBLE-DIFFERENCE LOCAL EARTHQUAKE TOMOGRAPHY (BlindG3VM-LET vs SIMULPS BASELINE)")
    print("=" * 120)

    forge_let = relocate_utah_forge_microseismic(num_events=16)
    print(
        f"[Scenario 05: Utah FORGE LET] Initial RMSE: {forge_let['initial_rmse_km']:.4f} km -> Relocated RMSE: {forge_let['blind_g3vm_let_rmse_km']:.4f} km | "
        f"Reduction over Straight: {forge_let['rmse_reduction_over_straight_pct']:.2f}% (>= 35% req) | Active Heap: {forge_let['blind_g3vm_heap_bytes'] / 1024:.1f} KB (< 2.0 MB)"
    )
    assert forge_let["rmse_reduction_over_straight_pct"] >= 35.0, f"Expected >=35% RMSE reduction in Utah FORGE, got {forge_let['rmse_reduction_over_straight_pct']:.2f}%"
    assert forge_let["blind_g3vm_heap_bytes"] < 2_000_000, f"Expected heap < 2.0 MB, got {forge_let['blind_g3vm_heap_bytes']} bytes"

    lunar_let = relocate_lunar_isru_moonquakes(num_events=8)
    print(
        f"[Scenario 06: Lunar ISRU LET] Initial RMSE: {lunar_let['initial_rmse_km']:.4f} km -> Relocated RMSE: {lunar_let['blind_g3vm_let_rmse_km']:.4f} km | "
        f"Reduction over Straight: {lunar_let['rmse_reduction_over_straight_pct']:.2f}% (>= 35% req) | Active Heap: {lunar_let['blind_g3vm_heap_bytes'] / 1024:.1f} KB (< 2.0 MB)"
    )
    assert lunar_let["rmse_reduction_over_straight_pct"] >= 35.0, f"Expected >=35% RMSE reduction in Lunar ISRU, got {lunar_let['rmse_reduction_over_straight_pct']:.2f}%"
    assert lunar_let["blind_g3vm_heap_bytes"] < 2_000_000, f"Expected heap < 2.0 MB, got {lunar_let['blind_g3vm_heap_bytes']} bytes"

    spherical_let = run_global_spherical_let_benchmark()
    print(
        f"[Scenario 07: Global Spherical LET] BlindG3VM-LET Smearing: {spherical_let['blind_g3vm_smearing']:.4f} vs "
        f"SIMULPS Smearing: {spherical_let['simulps_smearing']:.4f} | Heap: {spherical_let['blind_g3vm_heap_bytes'] / 1024:.1f} KB (< 2.0 MB)"
    )
    assert spherical_let["blind_g3vm_smearing"] < spherical_let["simulps_smearing"]
    assert spherical_let["blind_g3vm_heap_bytes"] < 2_000_000
    assert len(spherical_let["sample_phases"]) >= 4

    let_summary = {
        "engine": "G3VM (DarkSnail-Geo3D) v0.7.0",
        "utah_forge_egs_doublet": forge_let,
        "lunar_lava_tube_isru": lunar_let,
        "global_spherical_let_earth": spherical_let,
    }
    let_json_path = out_dir / "g3vm_v070_let_benchmark_summary.json"
    let_json_path.write_text(json.dumps(let_summary, indent=2), encoding="utf-8")

    eval_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bench_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit_json_path.write_text(json.dumps(audit_res, indent=2), encoding="utf-8")
    export_portable_html_studio(html_path, grid_shape=(24, 24, 24))
    showcase_png = out_dir / "g3vm_leapfrog_showcase.png"
    beltrami_png = out_dir / "g3vm_well_log_beltrami.png"
    render_leapfrog_showcase_figure(showcase_png, grid_shape=(32, 24, 24))
    render_well_log_beltrami_figure(beltrami_png, grid_shape=(48, 48, 48))

    print(f"[+] Saved evaluation summary : {eval_json_path}")
    print(f"[+] Saved benchmark summary  : {bench_json_path}")
    print(f"[+] Saved referee audit JSON : {audit_json_path}")
    print(f"[+] Saved LET summary JSON   : {let_json_path}")
    print(f"[+] Saved portable 3D studio : {html_path}")
    print(f"[+] Saved Mode B showcase    : {showcase_png}")
    print(f"[+] Saved Beltrami well logs : {beltrami_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
