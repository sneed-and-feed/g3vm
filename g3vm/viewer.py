"""
g3vm.viewer: Interactive 3D WebGL/Canvas Geological Workstation (Flask),
Zero-Install Portable HTML5 Studio Bundler, and High-Resolution Matplotlib
Publication Showcase Renderer.

Cross-platform compatible across Windows, macOS, and Linux (including PyInstaller
single-file binary bundles via `sys._MEIPASS`).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory

from .scenarios import (
    FAULT_MODE_NAMES,
    INDUSTRY_DEPLOYMENT_MATRIX,
    LITHOLOGY_CATALOG,
    SCENARIO_Descriptions,
    evaluate_scenario_comparison,
)


def resolve_static_dir() -> Path:
    """
    Resolve the `static/` directory across standard Python installs, editable checkouts,
    and PyInstaller frozen single-file executables (`sys._MEIPASS`) on Windows, macOS, and Linux.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        frozen_dir = Path(getattr(sys, "_MEIPASS")) / "g3vm" / "static"
        if frozen_dir.exists():
            return frozen_dir
    return Path(__file__).resolve().parent / "static"


def _tensor_to_nested_list(t: Any, decimals: int = 3) -> list[Any]:
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(np.float64), decimals)
    return arr.tolist()


def serialize_evaluation_payload(
    scenario_name: str = "subsalt_graben",
    grid_shape: tuple[int, int, int] = (32, 32, 32),
    tau_v: float = 0.20,
    triad_gain: float = 1.0,
    lambda_ridge: float = 1e-2,
    unroll_steps: int = 1,
) -> dict[str, Any]:
    """
    Evaluate a 3D geological scenario and serialize all 3D cubes, borehole logs,
    Jordan cavity masks, 4D continuity splitting fields, and theorem telemetry
    into a compact JSON-ready dictionary for the interactive 3D Web Studio.
    """
    from .blind_benchmark import evaluate_blind_holdout_and_baselines

    bundle = evaluate_scenario_comparison(
        scenario_name=scenario_name,
        grid_shape=grid_shape,
        tau_v=tau_v,
        triad_gain=triad_gain,
        lambda_ridge=lambda_ridge,
        unroll_steps=unroll_steps,
    )
    scen = bundle["scenario"]
    g3vm_out = bundle["g3vm_output"]
    naive_out = bundle["naive_output"]
    spec_out = bundle["spectral_output"]
    cont_out = bundle["continuity_output"]

    # Mode B (Zero Target Leakage, SNR=20dB) blind holdout & kriging baselines
    blind_bundle = evaluate_blind_holdout_and_baselines(
        scenario_name=scenario_name,
        grid_shape=grid_shape,
        snr_db=20.0,
        fault_jitter_voxels=0,
        return_cubes=True,
    )
    blind_cubes = blind_bundle["cubes"]
    blind_g3vm_vp = blind_cubes["g3vm_blind_vp"]
    krig_vp = blind_cubes["krig_vp"]
    fkrig_vp = blind_cubes["fkrig_vp"]
    blind_gain_3d = (krig_vp - scen["true_vp_cube"]).abs() - (
        blind_g3vm_vp - scen["true_vp_cube"]
    ).abs()

    triad_norm_3d = g3vm_out["d_ell_triad_3d"].norm(dim=-1)
    err_g3vm_3d = (g3vm_out["vp_cube_kms"] - scen["true_vp_cube"]).abs()
    err_naive_3d = (naive_out["vp_cube_kms"] - scen["true_vp_cube"]).abs()

    wells_payload = []
    for w in list(scen["wells"]) + list(scen.get("deviated_wells", [])):
        b = w["beltrami"]
        traj = w.get("trajectory")
        wells_payload.append(
            {
                "name": w["name"],
                "well_type": w.get("well_type", "vertical"),
                "y": int(w["y"]),
                "x": int(w["x"]),
                "traj_z": _tensor_to_nested_list(traj.z_vox, 2) if traj is not None else [],
                "traj_y": _tensor_to_nested_list(traj.y_vox, 2) if traj is not None else [],
                "traj_x": _tensor_to_nested_list(traj.x_vox, 2) if traj is not None else [],
                "max_inc_deg": round(float(np.max(traj.inc_deg)), 1) if traj is not None else 0.0,
                "azi_deg": round(float(np.mean(traj.azi_deg[-5:])), 1) if traj is not None else 0.0,
                "total_md_m": round(float(traj.md_m[-1]), 1) if traj is not None else 3000.0,
                "total_tvd_m": round(float(traj.tvd_m[-1]), 1) if traj is not None else 3000.0,
                "max_dls_deg_30m": round(float(np.max(traj.dogleg_deg_30m)), 2) if traj is not None else 0.0,
                "stringer_depths": [int(z) for z in w["stringer_depths"]],
                "true_vp": _tensor_to_nested_list(w["true_vp_trace"], 3),
                "reconstructed_vp": _tensor_to_nested_list(b.reconstructed_vp, 3),
                "naive_pooled_vp": _tensor_to_nested_list(b.naive_pooled_vp, 3),
                "salience": _tensor_to_nested_list(b.shielded_salience, 3),
                "cross_shell_curl": _tensor_to_nested_list(b.cross_shell_curl, 3),
                "tier1_indices": _tensor_to_nested_list(b.tier1_outlier_indices, 0),
                "tier1_values": _tensor_to_nested_list(b.tier1_outlier_values, 3),
                "tier2_indices": _tensor_to_nested_list(b.tier2_medoid_indices, 0),
                "tier2_values": _tensor_to_nested_list(b.tier2_medoid_values, 3),
                "adelic_gamma": _tensor_to_nested_list(b.adelic_gamma, 3),
                "beltrami_spike_recovery": round(float(b.beltrami_spike_recovery), 4),
                "naive_spike_recovery": round(float(b.naive_spike_recovery), 4),
                "all_stringers_recovery": round(float(getattr(b, "all_stringers_recovery", b.beltrami_spike_recovery)), 4),
                "equal_budget_naive_recovery": round(float(getattr(b, "equal_budget_naive_recovery", b.naive_spike_recovery)), 4),
                "thin_bed_f1": round(float(getattr(b, "thin_bed_f1", 1.0)), 4),
            }
        )

    return {
        "scenario_name": scen["scenario_name"],
        "meta": scen["meta"],
        "scenarios_available": SCENARIO_Descriptions,
        "industry_deployment_matrix": list(INDUSTRY_DEPLOYMENT_MATRIX),
        "lithology_catalog": list(scen.get("lithology_catalog", LITHOLOGY_CATALOG)),
        "fault_mode_names": list(FAULT_MODE_NAMES),
        "fault_planes_meta": scen["fault_planes_meta"],
        "grid_shape": list(scen["grid_shape"]),
        "params": {
            "tau_v": float(tau_v),
            "triad_gain": float(triad_gain),
            "lambda_ridge": float(lambda_ridge),
            "unroll_steps": int(unroll_steps),
        },
        "metrics": bundle["metrics"],
        "blind_metrics": {
            "snr_db": blind_bundle["snr_db"],
            "kriging_tuned_params": blind_bundle["kriging_tuned_params"],
            "lowo_heldout_well_rmse_kms": blind_bundle["lowo_heldout_well_rmse_kms"],
            "heldout_deviated_well_rmse_kms": blind_bundle["heldout_deviated_well_rmse_kms"],
            "whole_cube_blind_rmse_kms": blind_bundle["whole_cube_blind_rmse_kms"],
            "bounded_edge_gradient_fidelity": blind_bundle["bounded_edge_gradient_fidelity"],
        },
        "cubes": {
            "g3vm_vp": _tensor_to_nested_list(g3vm_out["vp_cube_kms"], 3),
            "g3vm_vp_tti": _tensor_to_nested_list(g3vm_out["vp_tti_cube_kms"], 3),
            "g3vm_vs": _tensor_to_nested_list(g3vm_out["vs_cube_kms"], 3),
            "g3vm_rho": _tensor_to_nested_list(g3vm_out["rho_cube_gcc"], 3),
            "g3vm_zp": _tensor_to_nested_list(g3vm_out["acoustic_impedance_zp"], 3),
            "g3vm_eps_r": _tensor_to_nested_list(g3vm_out["eps_r_cube"], 3),
            "g3vm_v_em": _tensor_to_nested_list(g3vm_out["v_em_cube_m_ns"], 4),
            "naive_eps_r": _tensor_to_nested_list(naive_out["eps_r_cube"], 3),
            "true_eps_r": _tensor_to_nested_list(scen["true_eps_r_cube"], 3),
            "twt_radar_us": _tensor_to_nested_list(g3vm_out["twt_radar_us_cube"], 3),
            "g3vm_epsilon": _tensor_to_nested_list(g3vm_out["epsilon_cube"], 3),
            "g3vm_delta": _tensor_to_nested_list(g3vm_out["delta_cube"], 3),
            "g3vm_gamma": _tensor_to_nested_list(g3vm_out["gamma_cube"], 3),
            "g3vm_vp_vs_ratio": _tensor_to_nested_list(g3vm_out["vp_vs_ratio_cube"], 3),
            "naive_vp": _tensor_to_nested_list(naive_out["vp_cube_kms"], 3),
            "true_vp": _tensor_to_nested_list(scen["true_vp_cube"], 3),
            "blind_g3vm_vp": _tensor_to_nested_list(blind_g3vm_vp, 3),
            "kriging_vp": _tensor_to_nested_list(krig_vp, 3),
            "fkriging_vp": _tensor_to_nested_list(fkrig_vp, 3),
            "blind_gain": _tensor_to_nested_list(blind_gain_3d, 3),
            "g3vm_litho": _tensor_to_nested_list(g3vm_out["lithology_cube"], 0),
            "naive_litho": _tensor_to_nested_list(naive_out["lithology_cube"], 0),
            "true_litho": _tensor_to_nested_list(scen["true_litho_cube"], 0),
            "triad_norm": _tensor_to_nested_list(triad_norm_3d, 3),
            "err_g3vm": _tensor_to_nested_list(err_g3vm_3d, 3),
            "err_naive": _tensor_to_nested_list(err_naive_3d, 3),
            "precision_pi": _tensor_to_nested_list(scen["precision_3d"], 3),
            "jordan_cavity": _tensor_to_nested_list(spec_out["cavity_info"]["jordan_6way"], 2),
            "adelic_gamma": _tensor_to_nested_list(spec_out["adelic_gamma_3d"], 3),
            "timelapse_nope_plume": _tensor_to_nested_list(
                cont_out["nope_plume_delta_3d"].abs(), 3
            ),
        },
        "wells": wells_payload,
    }


def export_portable_html_studio(
    output_path: str | Path = "studio/g3vm_portable_studio.html",
    grid_shape: tuple[int, int, int] = (24, 24, 24),
) -> Path:
    """
    Generate a 100% standalone, zero-install single-file HTML5 3D Studio
    (`studio/g3vm_portable_studio.html`) with pre-evaluated 3D cubes for all 6 scenarios
    embedded directly inside the HTML so users on Windows, macOS, Linux, or mobile
    can open it in any browser without Python or a backend server.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    template_path = resolve_static_dir() / "index.html"
    html_src = template_path.read_text(encoding="utf-8")

    embedded_scenarios = {}
    for scen_name in SCENARIO_Descriptions:
        embedded_scenarios[scen_name] = serialize_evaluation_payload(
            scenario_name=scen_name,
            grid_shape=grid_shape,
        )

    scenarios_json = json.dumps(
        embedded_scenarios,
        separators=(",", ":"),
        allow_nan=False,
    )
    embedded_script = (
        "<script>\n"
        "window.G3VM_PORTABLE_MODE = true;\n"
        f"window.G3VM_OFFLINE_SCENARIOS = /*G3VM_JSON_START*/{scenarios_json}/*G3VM_JSON_END*/;\n"
        "(function() {\n"
        "  const _origFetch = window.fetch ? window.fetch.bind(window) : null;\n"
        "  window.fetch = async function(input, init) {\n"
        "    const url = typeof input === 'string' ? input : (input && input.url ? input.url : '');\n"
        "    if (Boolean(window.G3VM_PORTABLE_MODE) && url.includes('/api/')) {\n"
        "      if (url.includes('/api/health')) {\n"
        "        return new Response(JSON.stringify({status:'ok', mode:'portable', version:'0.6.0'}), {\n"
        "          status: 200, headers: {'Content-Type': 'application/json'}\n"
        "        });\n"
        "      }\n"
        "      let bodyObj = {};\n"
        "      try { bodyObj = init && init.body ? JSON.parse(init.body) : {}; } catch (e) {}\n"
        "      const scen = bodyObj.scenario_name || 'subsalt_graben';\n"
        "      const src = (window.G3VM_OFFLINE_SCENARIOS && window.G3VM_OFFLINE_SCENARIOS[scen])\n"
        "        || Object.values(window.G3VM_OFFLINE_SCENARIOS || {})[0];\n"
        "      const payload = JSON.parse(JSON.stringify(src));\n"
        "      if (bodyObj.triad_gain === 0) {\n"
        "        payload.cubes.g3vm_vp = payload.cubes.naive_vp;\n"
        "        payload.cubes.err_g3vm = payload.cubes.err_naive;\n"
        "        if (payload.cubes.naive_eps_r) { payload.cubes.g3vm_eps_r = payload.cubes.naive_eps_r; }\n"
        "      }\n"
        "      return new Response(JSON.stringify(payload), {\n"
        "        status: 200, headers: {'Content-Type': 'application/json'}\n"
        "      });\n"
        "    }\n"
        "    return _origFetch ? _origFetch(input, init) : Promise.reject(new Error('fetch unavailable'));\n"
        "  };\n"
        "})();\n"
        "</script>\n"
    )

    patched_html = html_src.replace("</head>", f"{embedded_script}</head>", 1)
    out_file.write_text(patched_html, encoding="utf-8")
    return out_file


def create_studio_app(default_grid_shape: tuple[int, int, int] = (32, 32, 32)) -> Flask:
    """
    Create the Flask web server for the G3VM 3D Subsurface Velocity Studio.
    """
    static_dir = resolve_static_dir()
    app = Flask(__name__, static_folder=str(static_dir))
    app.config["DEFAULT_GRID_SHAPE"] = default_grid_shape

    @app.route("/")
    def index() -> Any:
        index_path = static_dir / "index.html"
        if index_path.exists():
            return send_from_directory(str(static_dir), "index.html")
        return render_template_string("<h1>G3VM Studio index.html missing</h1>"), 404

    @app.route("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "service": "g3vm-studio", "version": "0.6.0"})

    @app.route("/api/evaluate", methods=["GET", "POST"])
    def evaluate_endpoint() -> Any:
        data = request.get_json(silent=True) or {}
        scenario_name = str(
            data.get("scenario_name") or request.args.get("scenario_name") or "subsalt_graben"
        )
        grid_n = int(data.get("grid_size") or request.args.get("grid_size") or default_grid_shape[0])
        grid_n = max(16, min(64, grid_n))
        tau_v = float(
            data.get("tau_v") if "tau_v" in data else (request.args.get("tau_v") or 0.20)
        )
        triad_gain = float(
            data.get("triad_gain")
            if "triad_gain" in data
            else (request.args.get("triad_gain") or 1.0)
        )
        lambda_ridge = float(
            data.get("lambda_ridge")
            if "lambda_ridge" in data
            else (request.args.get("lambda_ridge") or 0.01)
        )
        unroll_steps = int(
            data.get("unroll_steps")
            if "unroll_steps" in data
            else (request.args.get("unroll_steps") or 1)
        )

        payload = serialize_evaluation_payload(
            scenario_name=scenario_name,
            grid_shape=(grid_n, grid_n, grid_n),
            tau_v=tau_v,
            triad_gain=triad_gain,
            lambda_ridge=lambda_ridge,
            unroll_steps=unroll_steps,
        )
        return jsonify(payload)

    return app


def render_leapfrog_showcase_figure(
    output_path: str | Path,
    grid_shape: tuple[int, int, int] = (32, 24, 24),
) -> dict[str, Any]:
    """
    Generate the high-DPI multi-panel Mode B (Zero Target Leakage, SNR = 20 dB) showcase figure
    comparing Ground Truth, Cross-Validated 3D Anisotropic Matérn-3/2 Kriging, and Blind
    DarkSnail G3VM across all six benchmark scenarios (including US DOE Utah FORGE and Lunar ISRU).
    """
    from .blind_benchmark import evaluate_blind_holdout_and_baselines

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    scenarios = [
        "subsalt_graben",
        "thrust_anticline",
        "unconformity_trap",
        "egs_tbm_hazard",
        "utah_forge_egs_doublet",
        "lunar_lava_tube_isru",
    ]
    bundles = {
        name: evaluate_blind_holdout_and_baselines(
            name,
            grid_shape=(32, 24, 24),
            snr_db=20.0,
            fault_jitter_voxels=0,
            return_cubes=True,
        )
        for name in scenarios
    }

    plt.style.use("dark_background")
    fig, axes = plt.subplots(len(scenarios), 4, figsize=(21, 26.0), dpi=160)
    fig.patch.set_facecolor("#0b0f17")

    for row_idx, name in enumerate(scenarios):
        b = bundles[name]
        cubes = b["cubes"]
        scen = cubes["scenario"]
        g3vm_vp = cubes["g3vm_blind_vp"].cpu().numpy()
        krig_vp = cubes["krig_vp"].cpu().numpy()
        true_vp = cubes["true_vp"].cpu().numpy()
        y_mid = true_vp.shape[1] // 2

        cw = b["whole_cube_blind_rmse_kms"]
        lw = b["lowo_heldout_well_rmse_kms"]
        dw = b["heldout_deviated_well_rmse_kms"]
        ef = b["bounded_edge_gradient_fidelity"]
        kp = b["kriging_tuned_params"]

        vp_vmin = 0.05 if name == "lunar_lava_tube_isru" else 1.7
        vp_vmax = 5.55 if name == "lunar_lava_tube_isru" else 6.4
        err_lim = 1.6 if name == "lunar_lava_tube_isru" else 1.1

        panels = [
            (
                true_vp[:, y_mid, :],
                f"{scen['meta']['title']}\n[Ground Truth Vp (km/s) @ Inline Y={y_mid}]",
                "turbo",
                vp_vmin,
                vp_vmax,
            ),
            (
                krig_vp[:, y_mid, :],
                f"Tuned Anisotropic Matérn-3/2 Kriging (lz={kp['ell_z']:.0f}, lxy={kp['ell_xy']:.0f})\nCube RMSE: {cw['tuned_anisotropic_kriging']:.3f} | LOWO Well: {lw['tuned_anisotropic_kriging']:.3f} | EdgeFid: {ef['tuned_anisotropic_kriging']*100:.1f}%",
                "turbo",
                vp_vmin,
                vp_vmax,
            ),
            (
                g3vm_vp[:, y_mid, :],
                f"Blind DarkSnail G3VM (Mode B, SNR=20dB, Zero Leakage)\nCube RMSE: {cw['blind_darksnail_g3vm']:.3f} | LOWO Well: {lw['blind_darksnail_g3vm']:.3f} | EdgeFid: {ef['blind_darksnail_g3vm']*100:.1f}%",
                "turbo",
                vp_vmin,
                vp_vmax,
            ),
            (
                np.abs(krig_vp[:, y_mid, :] - true_vp[:, y_mid, :])
                - np.abs(g3vm_vp[:, y_mid, :] - true_vp[:, y_mid, :]),
                f"Signed Blind Error Gain (|Err_Kriging| - |Err_BlindG3VM|)\nDevWell RMSE: {dw['blind_darksnail_g3vm']:.3f} vs Krig {dw['tuned_anisotropic_kriging']:.3f} | FKrig Cube: {cw['fault_partitioned_kriging']:.3f}",
                "coolwarm",
                -err_lim,
                err_lim,
            ),
        ]

        for col_idx, (img, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#111726")
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_title(title, fontsize=8.8, fontweight="bold", color="#e6edf3", pad=6)
            ax.set_xlabel("Crossline X (East)", fontsize=8.0, color="#9da7b3")
            ax.set_ylabel("Depth Z (Vox)", fontsize=8.0, color="#9da7b3")
            ax.tick_params(colors="#8b949e", labelsize=7.5)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=7.0, colors="#8b949e")

            if col_idx in (0, 2):
                for w in scen["wells"]:
                    ax.axvline(w["x"], color="#ffffff", linestyle="--", linewidth=0.85, alpha=0.65)
                    for sz in w["stringer_depths"]:
                        ax.plot(
                            w["x"],
                            sz,
                            marker="o",
                            markersize=4.2,
                            color="#00f5d4",
                            markeredgecolor="#000000",
                        )

    fig.suptitle(
        "G3VM v0.6.0 Mode B Blind Benchmark (SNR=20dB, Zero Target Leakage): Ground Truth vs. CV-Tuned Matérn-3/2 Kriging vs. Blind G3VM",
        fontsize=12.5,
        fontweight="bold",
        color="#58a6ff",
        y=0.992,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(out_file, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {name: {k: v for k, v in bundles[name].items() if k != "cubes"} for name in scenarios}


def render_well_log_beltrami_figure(
    output_path: str | Path,
    grid_shape: tuple[int, int, int] = (48, 48, 48),
) -> dict[str, Any]:
    """
    Generate the 6-borehole sonic log & pure non-circular Beltrami exterior-curl figure under
    20 dB sonic noise and 10% borehole caliper washout dropouts, comparing against an
    Equal-Token-Budget (K=16 tokens) Uniform Pooler and Coarse Seismic-Grid (Z//8) Pooler.
    """
    import torch

    from .blind_benchmark import _add_deterministic_noise
    from .engine import BeltramiWellLogShell
    from .scenarios import build_geological_scenario

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    scen_sub = build_geological_scenario("subsalt_graben", grid_shape=grid_shape)
    scen_forge = build_geological_scenario("utah_forge_egs_doublet", grid_shape=grid_shape)
    scen_lunar = build_geological_scenario("lunar_lava_tube_isru", grid_shape=(32, 24, 24))

    selected_wells = [
        scen_sub["wells"][0],
        scen_forge["deviated_wells"][0],
        scen_forge["deviated_wells"][1],
        scen_lunar["wells"][0],
        scen_lunar["deviated_wells"][0],
        scen_lunar["deviated_wells"][1],
    ]

    shell = BeltramiWellLogShell(outlier_budget=8, condensed_budget=8)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, len(selected_wells), figsize=(23, 7.8), dpi=160)
    fig.patch.set_facecolor("#0b0f17")

    out_summary: dict[str, Any] = {}

    for idx, w in enumerate(selected_wells):
        ax = axes[idx]
        ax.set_facecolor("#111726")
        clean_t = w["true_vp_trace"].float()
        Z_w = clean_t.numel()
        depths = np.arange(Z_w)

        # Apply honest 20 dB sonic noise + 10% caliper washout dropout
        noisy_t = _add_deterministic_noise(clean_t, snr_db=20.0, seed=500 + idx)
        gen = torch.Generator()
        gen.manual_seed(700 + idx)
        perm = torch.randperm(Z_w - 4, generator=gen) + 2
        n_drop = max(1, int(round(0.10 * Z_w)))
        drop_mask = torch.zeros(Z_w, dtype=torch.bool)
        # Ensure we test realistic random caliper washouts
        drop_mask[perm[:n_drop]] = True
        valid_z = torch.nonzero(~drop_mask, as_tuple=False).view(-1).float()
        valid_v = noisy_t[~drop_mask]
        obs_t = torch.tensor(
            np.interp(np.arange(Z_w, dtype=np.float32), valid_z.numpy(), valid_v.numpy()),
            dtype=torch.float32,
        )

        b = shell(obs_t, reference_trace=clean_t)
        eq_bins = min(max(4, 16), max(4, Z_w // 2))
        eq_down = torch.nn.functional.adaptive_avg_pool1d(obs_t.view(1, 1, Z_w), eq_bins)
        eq_up = (
            torch.nn.functional.interpolate(eq_down, size=Z_w, mode="linear", align_corners=False)
            .view(-1)
            .cpu()
            .numpy()
        )

        true_tr = clean_t.cpu().numpy()
        obs_tr = obs_t.cpu().numpy()
        rec_tr = b.reconstructed_vp.cpu().numpy()
        naive_tr = b.naive_pooled_vp.cpu().numpy()
        salience = b.shielded_salience.cpu().numpy()
        t1_idx = b.tier1_outlier_indices.cpu().numpy().astype(int)
        t2_idx = b.tier2_medoid_indices.cpu().numpy().astype(int)
        t2_vals = b.tier2_medoid_values.cpu().numpy()

        # Shade caliper washout intervals
        for dz_i in torch.nonzero(drop_mask, as_tuple=False).view(-1).tolist():
            ax.axhspan(dz_i - 0.45, dz_i + 0.45, color="#ff5470", alpha=0.16)

        ax.plot(true_tr, depths, color="#58a6ff", linewidth=1.7, alpha=0.85, label="Clean Ground Truth Vp")
        ax.plot(obs_tr, depths, color="#adb5bd", linestyle=":", linewidth=1.1, label="Noisy+Washout Input (20dB)")
        ax.plot(
            eq_up,
            depths,
            color="#ff9f1c",
            linestyle="-.",
            linewidth=1.5,
            label=f"Equal-Budget K=16 Pool ({b.equal_budget_naive_recovery*100:.1f}%)",
        )
        ax.plot(
            naive_tr,
            depths,
            color="#ff6b6b",
            linestyle="--",
            linewidth=1.3,
            label=f"Coarse Z//8 Pool ({b.naive_spike_recovery*100:.1f}%)",
        )
        ax.plot(
            rec_tr,
            depths,
            color="#00f5d4",
            linewidth=1.8,
            label=f"Beltrami K=16 (All-Spikes: {b.all_stringers_recovery*100:.1f}%, F1={b.thin_bed_f1:.2f})",
        )
        ax.scatter(
            obs_tr[t1_idx],
            t1_idx,
            color="#ffd43b",
            s=30,
            zorder=5,
            edgecolors="#000000",
            label="Tier-1 Spikes (Pure Curl)",
        )
        ax.scatter(
            t2_vals,
            t2_idx,
            color="#c77dff",
            marker="s",
            s=22,
            zorder=4,
            label="Tier-2 Medoids",
        )

        ax_top = ax.twiny()
        ax_top.fill_betweenx(
            depths, 0.0, salience, color="#00f5d4", alpha=0.12
        )
        ax_top.set_xlim(0.0, 1.35)
        ax_top.tick_params(labelsize=6.8, colors="#00f5d4")
        ax_top.set_xlabel("Beltrami Exterior Curl", fontsize=7.2, color="#00f5d4")

        ax.set_title(
            f"{w['name']}\n[20dB, 10% Washout | Rec: {b.all_stringers_recovery*100:.1f}% | F1: {b.thin_bed_f1:.2f}]",
            fontsize=7.6,
            fontweight="bold",
            color="#e6edf3",
            pad=10,
        )
        ax.set_xlabel("Vp (km/s)", fontsize=8.5, color="#9da7b3")
        if idx == 0:
            ax.set_ylabel("Depth Station Index Z (Shallow -> Deep)", fontsize=9.0, color="#9da7b3")
        ax.invert_yaxis()
        ax.grid(True, color="#21262d", linestyle=":", linewidth=0.7)
        ax.legend(loc="lower left", fontsize=6.2, framealpha=0.88, facecolor="#0d1117")

        out_summary[w["name"]] = {
            "beltrami_spike_recovery": float(b.beltrami_spike_recovery),
            "all_stringers_recovery": float(b.all_stringers_recovery),
            "equal_budget_naive_recovery": float(b.equal_budget_naive_recovery),
            "naive_spike_recovery": float(b.naive_spike_recovery),
            "thin_bed_f1": float(b.thin_bed_f1),
        }

    fig.suptitle(
        "G3VM v0.6.0 Pure Non-Circular Beltrami Shell (K=8+8=16 Tokens, SNR=20dB, 10% Caliper Washout) vs. Equal-Budget (K=16) & Coarse Pooling",
        fontsize=12.2,
        fontweight="bold",
        color="#00f5d4",
        y=0.985,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.945])
    fig.savefig(out_file, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_summary
