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
    ThomsenTTIElasticModulusEngine,
    ZenoSurgicalAnomalyGate,
    build_geological_scenario,
    build_kelvin_mandel_so6_rotor,
    build_so3_from_dip_azimuth,
    build_vti_kelvin_stiffness_tensor,
    compute_marginal_wasserstein1_vp,
    compute_tti_phase_velocities,
    create_studio_app,
    evaluate_scenario_comparison,
    export_portable_html_studio,
    read_dev_survey,
    read_las2_file,
    read_segy_cube,
    resolve_scenario_name,
    resolve_static_dir,
    solve_minimum_curvature_trajectory,
    write_dev_survey,
    write_las2_file,
    write_segy_cube,
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
    assert len(payload["wells"]) == 8  # 6 vertical + 2 deviated/horizontal wells
    assert "g3vm_vp_tti" in payload["cubes"]
    assert "g3vm_zp" in payload["cubes"]
    assert "g3vm_vs" in payload["cubes"]
    assert "g3vm_rho" in payload["cubes"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        html_out = export_portable_html_studio(tmp / "portable.html", grid_shape=(16, 16, 16))
        assert html_out.exists()
        html_txt = html_out.read_text(encoding="utf-8")
        assert "window.G3VM_PORTABLE_MODE = true;" in html_txt
        assert "window.G3VM_OFFLINE_SCENARIOS" in html_txt
        assert "window.fetch = async function" in html_txt
        for req_key in (
            '"g3vm_vp_tti"',
            '"g3vm_zp"',
            '"g3vm_vs"',
            '"g3vm_rho"',
            '"g3vm_epsilon"',
            '"g3vm_delta"',
            '"g3vm_vp_vs_ratio"',
            '"Well-G7 (Deviated Subsalt S-Curve)"',
            '"Well-H8 (Horizontal EGS Lateral)"',
            '"traj_z"',
            '"traj_y"',
            '"traj_x"',
        ):
            assert req_key in html_txt

        npz_out = tmp / "cube.npz"
        rc = cli_main(["export", "--scenario", "subsalt_graben", "--grid-size", "16", "--output", str(npz_out)])
        assert rc == 0
        assert npz_out.exists()
        with np.load(npz_out) as loaded:
            assert loaded["g3vm_vp_kms"].shape == (16, 16, 16)
            assert loaded["g3vm_vp_tti_kms"].shape == (16, 16, 16)
            assert loaded["g3vm_vs_kms"].shape == (16, 16, 16)


def test_suite9_deviated_horizontal_wells_and_las2_segy_io():
    """
    Verify SPE Minimum Curvature 3D trajectory integration on deviated/horizontal wells
    and bit-accurate round-trip I/O for CWLS LAS 2.0 (.las), .dev surveys, and SEG-Y Rev 1 (.segy).
    """
    S = 48
    md = np.linspace(0.0, 3800.0, S, dtype=np.float64)
    inc = np.clip(np.linspace(0.0, 120.0, S), 0.0, 88.0)  # Horizontal lateral
    azi = np.full(S, 75.0, dtype=np.float64)

    traj = solve_minimum_curvature_trajectory(
        md_m=md,
        inc_deg=inc,
        azi_deg=azi,
        surface_yx_vox=(16.0, 8.0),
        grid_shape=(32, 32, 32),
        well_name="Well-H8-Horizontal",
        well_type="horizontal_lateral",
    )
    assert traj.tvd_m[-1] < traj.md_m[-1] * 0.75
    assert traj.east_x_m[-1] > traj.east_x_m[0] + 1000.0
    assert np.all(traj.dogleg_deg_30m >= 0.0)

    vp_true = np.full(S, 3.2, dtype=np.float64)
    vp_true[18] = 5.55
    vp_true[34] = 5.80

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. CWLS LAS 2.0 round-trip + CLI ingest-las
        las_path = write_las2_file(tmp / "well_h8.las", trajectory=traj, vp_kms=vp_true)
        assert las_path.exists()
        rec = read_las2_file(las_path, grid_shape=(32, 32, 32))
        assert rec.well_name == "Well-H8-Horizontal"
        assert np.allclose(rec.curves["VP"], vp_true, atol=1e-3)
        assert np.allclose(rec.trajectory.inc_deg, inc, atol=1e-2)

        shielded_las = tmp / "well_h8_shielded.las"
        rc_las = cli_main(["ingest-las", str(las_path), "--output-las", str(shielded_las)])
        assert rc_las == 0
        assert shielded_las.exists()

        # 2. Directional survey (.dev) round-trip
        dev_path = write_dev_survey(tmp / "well_h8.dev", traj)
        traj_loaded = read_dev_survey(dev_path, grid_shape=(32, 32, 32))
        assert np.allclose(traj_loaded.tvd_m, traj.tvd_m, atol=1e-2)
        assert np.allclose(traj_loaded.x_vox, traj.x_vox, atol=1e-2)

        # 3. SEG-Y Rev 1 3D binary cube round-trip + CLI ingest-segy
        cube_orig = np.linspace(1.8, 6.2, 16 * 12 * 14, dtype=np.float32).reshape(16, 12, 14)
        segy_path = write_segy_cube(tmp / "velocity_tti.segy", cube_orig, sample_interval_us=2000)
        assert segy_path.exists()
        segy_loaded = read_segy_cube(segy_path)
        assert segy_loaded["cube_zyx"].shape == (16, 12, 14)
        assert segy_loaded["sample_interval_us"] == 2000
        assert segy_loaded["format_code"] == 5
        assert np.allclose(segy_loaded["cube_zyx"], cube_orig, atol=1e-6)

        rc_segy = cli_main(["ingest-segy", str(segy_path)])
        assert rc_segy == 0


def test_suite10_thomsen_tti_anisotropy_and_so6_kelvin_stiffness_rotor():
    """
    Verify Theorem 2.7 (so(6) Kelvin-Mandel stiffness rotor isometry) and Thomsen TTI
    phase velocity limits across tilted shale/carbonate strata.
    """
    dip = torch.tensor([0.0, 25.0, 48.0, 72.0], dtype=torch.float64)
    azi = torch.tensor([0.0, 45.0, 130.0, 260.0], dtype=torch.float64)
    R3 = build_so3_from_dip_azimuth(dip, azi)
    M6 = build_kelvin_mandel_so6_rotor(R3)

    eye6 = torch.eye(6, dtype=torch.float64).unsqueeze(0)
    ortho_err = float(torch.max(torch.abs(torch.matmul(M6.transpose(-1, -2), M6) - eye6)).item())
    assert ortho_err <= 1e-14

    C_vti = build_vti_kelvin_stiffness_tensor(
        vp0_kms=torch.full((4,), 3.2, dtype=torch.float64),
        vs0_kms=torch.full((4,), 1.65, dtype=torch.float64),
        rho_gcc=torch.full((4,), 2.35, dtype=torch.float64),
        epsilon=torch.full((4,), 0.22, dtype=torch.float64),
        delta=torch.full((4,), 0.09, dtype=torch.float64),
        gamma=torch.full((4,), 0.20, dtype=torch.float64),
    )
    C_tti = torch.matmul(torch.matmul(M6, C_vti), M6.transpose(-1, -2))

    # Eigenvalues (and therefore trace, Frobenius norm, and positive definiteness) must be invariant
    evals_vti = torch.linalg.eigvalsh(C_vti)
    evals_tti = torch.linalg.eigvalsh(C_tti)
    assert float(torch.max(torch.abs(evals_vti - evals_tti)).item()) <= 1e-12
    assert float(evals_tti.min().item()) > 0.0

    # Check exact Thomsen phase velocity limits at psi = 0 (symmetry axis) and psi = 90 deg (bedding plane)
    vp0 = torch.tensor([3.0], dtype=torch.float32)
    vs0 = torch.tensor([1.5], dtype=torch.float32)
    eps = torch.tensor([0.20], dtype=torch.float32)
    dlt = torch.tensor([0.08], dtype=torch.float32)
    gam = torch.tensor([0.16], dtype=torch.float32)
    zero_ang = torch.tensor([0.0], dtype=torch.float32)

    # Propagation parallel to vertical symmetry axis (psi = 0): Vp(0) == Vp0, Vsh(0) == Vs0
    vp_0, vsv_0, vsh_0 = compute_tti_phase_velocities(
        vp0, vs0, eps, dlt, gam, zero_ang, zero_ang, ray_direction_zyx=(1.0, 0.0, 0.0)
    )
    assert float(vp_0.item()) == pytest.approx(3.0, abs=1e-6)
    assert float(vsv_0.item()) == pytest.approx(1.5, abs=1e-6)
    assert float(vsh_0.item()) == pytest.approx(1.5, abs=1e-6)

    # Propagation parallel to bedding plane (psi = 90 deg): Vp(90) == Vp0*(1+eps), Vsh(90) == Vs0*(1+gam)
    vp_90, _, vsh_90 = compute_tti_phase_velocities(
        vp0, vs0, eps, dlt, gam, zero_ang, zero_ang, ray_direction_zyx=(0.0, 0.0, 1.0)
    )
    assert float(vp_90.item()) == pytest.approx(3.0 * 1.20, abs=1e-5)
    assert float(vsh_90.item()) == pytest.approx(1.5 * 1.16, abs=1e-5)

