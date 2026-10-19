"""
tests/test_g3vm.py: Comprehensive verification and bug-audit suite for G3VM (`DarkSnail-Geo3D`).
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from g3vm import (
    BeltramiWellLogShell,
    ContinuousSpectralGeoField3D,
    DarkSnailGeo3DVelocityModeler,
    NaiveSmoothVelocityBaseline,
    RoPE3DMedoidPooler,
    SlowFastGeoCascade,
    TemporalSurveyContinuitySplitter,
    ZenoSurgicalAnomalyGate,
    build_geological_scenario,
    compute_marginal_wasserstein1_vp,
    create_studio_app,
    evaluate_scenario_comparison,
    export_portable_html_studio,
    resolve_scenario_name,
    resolve_static_dir,
)
from g3vm.cli import main as cli_main


def test_suite1_drop_in_skeleton_contract_and_autograd():
    """Verify exact drop-in API compatibility with the DarkSnailGeo3DVelocityModeler specification."""
    model = DarkSnailGeo3DVelocityModeler(
        grid_shape=(12, 12, 12),
        num_lithologies=8,
        num_fault_modes=6,
    )
    nn_init = torch.randn_like(model.W_triad)
    model.W_triad.data.copy_(nn_init)

    ell_sym = torch.randn(12, 12, 12, 8, requires_grad=True)
    u_fault = torch.rand(12, 12, 12, 6)

    out = model(ell_sym, u_fault, tau_v=0.25)
    assert out["vp_cube_kms"].shape == (12, 12, 12)
    assert out["lithology_cube"].shape == (12, 12, 12)
    assert out["max_orthogonality_residual"] <= 1e-8

    loss = out["vp_cube_kms"].sum()
    loss.backward()
    assert model.W_triad.grad is not None
    assert torch.isfinite(model.W_triad.grad).all()


def test_suite2_formal_theorems_2_1_to_2_6():
    """Verify machine-checked invariants from OcclusionTriad.lean (Theorems 2.1 through 2.6)."""
    model = DarkSnailGeo3DVelocityModeler(grid_shape=(16, 16, 16))
    audit = model.verify_formal_theorems()

    assert audit["theorem_2_1_antisymmetry_error"] == 0.0
    assert audit["theorem_2_1_self_interaction_error"] == 0.0
    assert audit["theorem_2_2_tie_break_gap"] > 0.0
    assert audit["theorem_2_2_formula_error"] < 1e-6
    assert audit["theorem_2_3_orthogonality_residual"] <= 1e-8
    assert audit["theorem_2_4_step0_homotopy_error"] == 0.0
    assert audit["theorem_2_5_relative_phase_error"] < 1e-12
    assert audit["theorem_2_5_arithmetic_norm_loss"] > 0.10
    assert audit["theorem_2_6_spherical_radial_drift"] <= 1e-8
    assert audit["theorem_2_6_cascade_orthogonality_error"] <= 1e-8
    assert audit["all_theorems_verified"] is True


def test_suite3_zero_fault_context_and_degenerate_inputs():
    """Audit edge cases: zero fault indicators, zero vectors, and constant borehole traces."""
    model = DarkSnailGeo3DVelocityModeler(grid_shape=(8, 8, 8))
    model.W_triad.data.normal_(0.0, 1.0)

    # Zero u_fault must produce zero triad perturbation everywhere
    u_zero = torch.zeros(64, 6)
    v_probs = torch.softmax(torch.randn(64, 8), dim=-1)
    d_ell = model.skew_symmetric_fault_triad(u_zero, v_probs)
    assert float(d_ell.abs().max().item()) == 0.0

    # Zero v_litho must not NaN
    v_zero = torch.zeros(16, 8)
    u_rand = torch.randn(16, 6)
    d_ell_z = model.skew_symmetric_fault_triad(u_rand, v_zero)
    assert torch.isfinite(d_ell_z).all()

    # Constant borehole trace must not NaN
    const_trace = torch.full((64,), 3.0)
    salience = model.beltrami_well_log_salience(const_trace)
    assert salience.shape == (64,)
    assert torch.isfinite(salience).all()
    assert float(salience.max().item()) == 0.0


def test_suite4_beltrami_well_log_spike_recovery_vs_naive():
    """Verify BeltramiWellLogShell preserves 1-voxel high-contrast stringers at 1.0000 recovery."""
    Z = 64
    trace = torch.full((Z,), 2.4, dtype=torch.float32)
    trace[20] = 5.4  # Thin 1-voxel carbonate stringer
    trace[44] = 5.8  # Thin 1-voxel dolomite stringer

    shell = BeltramiWellLogShell(outlier_budget=8, condensed_budget=8)
    res = shell(trace)

    assert res.beltrami_spike_recovery == pytest.approx(1.0, abs=1e-6)
    assert res.naive_spike_recovery < 0.35
    assert 20 in res.tier1_outlier_indices.tolist()
    assert 44 in res.tier1_outlier_indices.tolist()
    assert float(res.reconstructed_vp[20].item()) == pytest.approx(5.4, abs=1e-5)
    assert float(res.reconstructed_vp[44].item()) == pytest.approx(5.8, abs=1e-5)
    assert bool(((res.adelic_gamma >= 0.05) & (res.adelic_gamma <= 1.0)).all().item())


def test_suite5_spectral_field_and_3d_jordan_cavity_isolation():
    """Verify ContinuousSpectralGeoField3D isolates enclosed 3D cavities from open exterior."""
    Z, Y, X = 24, 24, 24
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z, dtype=torch.float32),
        torch.arange(Y, dtype=torch.float32),
        torch.arange(X, dtype=torch.float32),
        indexing="ij",
    )
    r2 = ((zz - 12.0) / 6.0) ** 2 + ((yy - 12.0) / 6.0) ** 2 + ((xx - 12.0) / 6.0) ** 2
    hollow_shell = ((r2 >= 0.55) & (r2 <= 1.05)).float()

    spec = ContinuousSpectralGeoField3D()
    strat = 2.5 + 0.5 * (zz / Z)
    out = spec(strat, hollow_shell)

    cav = out["cavity_info"]
    j6 = cav["jordan_6way"]
    # Center (12, 12, 12) must be detected as enclosed 6-way cavity
    assert float(j6[12, 12, 12].item()) == 1.0
    # Exterior corners must NOT be detected as enclosed cavity
    assert float(j6[0, 0, 0].item()) == 0.0
    assert float(j6[Z - 1, Y - 1, X - 1].item()) == 0.0
    assert cav["medoid_zyx"] == (12, 12, 12)

    gamma = out["adelic_gamma_3d"]
    assert bool(((gamma >= 0.05) & (gamma <= 1.0)).all().item())


def test_suite6_ridge_omp_tomography_continuity_and_surgical_gate():
    """Verify Precision-Weighted Ridge + OMP tomography, W1_marg, and 4D H_clock (+) H_nope splitter."""
    model = DarkSnailGeo3DVelocityModeler(grid_shape=(16, 16, 16))
    N_vox = 16 * 16 * 16
    M_basis = 8
    D = torch.randn(N_vox, M_basis)
    true_w = torch.tensor([0.3, -0.25, 0.0, 0.15, 0.0, 0.0, -0.1, 0.0])
    pi = torch.rand(N_vox, 1) + 0.2
    r_true = (D @ true_w).unsqueeze(-1)

    upd_ridge = model.solve_ridge_tomography_update(D, r_true, pi, lambda_ridge=1e-4)
    mse_before = float((r_true.squeeze(-1) ** 2).mean().item())
    mse_after = float(((r_true.squeeze(-1) - upd_ridge) ** 2).mean().item())
    assert mse_after < 1e-3 * mse_before

    omp_res = model.solve_abelian_omp_tomography(D, r_true, pi, lambda_ridge=1e-4, max_atoms=6)
    assert omp_res["weighted_residual_mse"] < 1e-3 * mse_before

    # Marginal Wasserstein-1 check
    cube_a = torch.full((16, 16, 16), 3.0)
    cube_b = torch.full((16, 16, 16), 4.2)
    assert compute_marginal_wasserstein1_vp(cube_a, cube_a) == pytest.approx(0.0, abs=1e-9)
    assert compute_marginal_wasserstein1_vp(cube_a, cube_b) > 0.10

    # 4D Continuity Splitter check
    splitter = TemporalSurveyContinuitySplitter()
    mon1 = cube_a.clone()
    mon1[:3] += 0.04
    mon1[8:11, 8:11, 8:11] -= 0.25
    mon2 = cube_a.clone()
    mon2[:3] -= 0.04
    mon2[8:11, 8:11, 8:11] -= 0.35
    split_res = splitter.split_4d_surveys(cube_a, [mon1, mon2])
    assert split_res["exogenous_drift_invariance"] is True
    assert float(split_res["nope_plume_delta_3d"][:2].abs().max().item()) == 0.0
    assert float(split_res["nope_plume_delta_3d"][9, 9, 9].item()) == pytest.approx(-0.35, abs=1e-5)

    # Surgical gate check
    gate = ZenoSurgicalAnomalyGate(surgical_voxel_budget=16)
    gate_res = gate(split_res["nope_plume_delta_3d"])
    assert gate_res["scheduled_unroll_K"] in (1, 4, 8, 16)
    assert int(gate_res["surgical_mask_3d"].sum().item()) == 16


@pytest.mark.parametrize(
    "scenario_name",
    ["marmousi_subsalt", "seg_eage_overthrust", "ccs_h2_timelapse", "egs_tbm_hazard"],
)
def test_suite7_benchmark_scenarios_outperform_naive_rbf(scenario_name: str):
    """Verify G3VM strictly outperforms Naive RBF/Kriging across all 4 geophysical benchmarks."""
    bundle = evaluate_scenario_comparison(scenario_name, grid_shape=(24, 24, 24))
    m = bundle["metrics"]

    assert m["num_sparse_boreholes"] == 6
    assert m["g3vm_vp_rmse_kms"] < m["naive_vp_rmse_kms"]
    assert m["g3vm_fault_zone_rmse_kms"] < m["naive_fault_zone_rmse_kms"]
    assert m["g3vm_fault_sharpness_ratio"] > m["naive_fault_sharpness_ratio"]
    assert m["g3vm_lithology_accuracy"] > m["naive_lithology_accuracy"]
    assert m["g3vm_marginal_w1"] < m["naive_marginal_w1"]
    assert m["beltrami_spike_recovery"] == pytest.approx(1.0, abs=1e-4)
    assert m["beltrami_spike_recovery"] > m["naive_spike_recovery"]
    assert m["max_orthogonality_residual"] <= 1e-8
    assert m["step0_homotopy_error"] == 0.0


def test_suite8_web_studio_endpoints_and_portable_bundle():
    """Verify Flask Studio endpoints, portable HTML5 bundle generation, and CLI export."""
    assert (resolve_static_dir() / "index.html").exists()
    assert resolve_scenario_name("marmousi_subsalt") == "subsalt_graben"

    app = create_studio_app(default_grid_shape=(16, 16, 16))
    client = app.test_client()

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.get_json()["status"] == "ok"

    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert b"G3VM Studio" in r_idx.data

    r_eval = client.post(
        "/api/evaluate",
        json={"scenario_name": "seg_eage_overthrust", "grid_size": 16, "triad_gain": 1.0},
    )
    assert r_eval.status_code == 200
    payload = r_eval.get_json()
    assert payload["scenario_name"] == "thrust_anticline"
    assert payload["grid_shape"] == [16, 16, 16]
    assert len(payload["wells"]) == 6

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        html_out = export_portable_html_studio(tmp / "portable.html", grid_shape=(16, 16, 16))
        assert html_out.exists()
        assert "G3VM_OFFLINE_SCENARIOS" in html_out.read_text(encoding="utf-8")

        npz_out = tmp / "cube.npz"
        rc = cli_main(["export", "--scenario", "subsalt_graben", "--grid-size", "16", "--output", str(npz_out)])
        assert rc == 0
        assert npz_out.exists()
        with np.load(npz_out) as loaded:
            assert loaded["g3vm_vp_kms"].shape == (16, 16, 16)
