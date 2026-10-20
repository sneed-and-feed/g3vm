"""
G3VM v0.3.0 Verification Suite:
1. Out-of-Core SEG-Y Rev 1 Streaming (`np.memmap`) & Beltrami Exterior-Curl Downsampling
2. Z-Slab Chunked 3D Spectral & Triad Evaluation (`chunk_z` Bit-Identity & SO(6) Orthogonality)
3. US DOE Utah FORGE EGS Doublet (`Well 16A(78)-32` & `Well 16B(78)-32`) Spike Recovery & Fault RMSE
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from g3vm.cli import main as cli_main
from g3vm.engine import BeltramiWellLogShell, DarkSnailGeo3DVelocityModeler
from g3vm.io import (
    ensure_utah_forge_field_files,
    read_dev_survey,
    read_las2_file,
    stream_segy_volume,
    write_segy_cube,
)
from g3vm.scenarios import (
    FAULT_MODE_NAMES,
    LITHOLOGY_CATALOG,
    build_geological_scenario,
    evaluate_scenario_comparison,
)


def test_segy_memmap_out_of_core_windowing(tmp_path: Path):
    """
    Synthesize a 400 x 400 x 100 (64 MB) binary SEG-Y Rev 1 file on disk, stream a
    64 x 64 x 64 ROI window via `stream_segy_volume()`, verify peak Python heap allocation
    (`tracemalloc`) stays < 35 MB, and assert 100% fault step jump recovery vs. naive block averaging.
    """
    n_il, n_xl, n_z = 400, 400, 100
    zz = np.arange(n_z, dtype=np.float32)[:, None, None]
    yy = np.arange(n_il, dtype=np.float32)[None, :, None]
    xx = np.arange(n_xl, dtype=np.float32)[None, None, :]

    fault_plane = xx - (180.0 + 0.25 * zz + 0.05 * (yy - 200.0))
    cube = np.where(fault_plane >= 0.0, 5.82, 2.65).astype(np.float32)
    cube[48, :, :] = 3.70

    segy_path = tmp_path / "utah_forge_large_400x400x100.segy"
    write_segy_cube(
        segy_path,
        cube,
        inline_start=1000,
        crossline_start=2000,
        sample_interval_us=2000,
        format_code=5,
    )
    del cube

    assert segy_path.exists()
    assert segy_path.stat().st_size >= 64_000_000  # > 64 MB on disk

    # 1. Stream a 64 x 64 x 64 ROI window covering the fault plane and thin-bed stringer
    roi_spec = "1150:1213,2150:2213,16:80"
    res_roi = stream_segy_volume(
        segy_path,
        roi=roi_spec,
        max_grid=64,
    )

    assert res_roi["raw_shape_zyx"] == (100, 400, 400)
    assert res_roi["roi_shape_zyx"] == (64, 64, 64)
    assert res_roi["downsampled_shape_zyx"] == (64, 64, 64)
    assert res_roi["peak_ram_mb"] < 35.0, f"Peak RAM {res_roi['peak_ram_mb']:.2f} MB exceeded 35 MB limit!"
    assert res_roi["fault_step_recovery"] == pytest.approx(1.0, abs=1e-4)

    # 2. Stream a strided 128 x 128 x 64 window downsampled to 32 x 32 x 32 via Beltrami exterior curl
    res_ds = stream_segy_volume(
        segy_path,
        roi="1120:1247,2120:2247,16:80",
        max_grid=32,
    )
    assert res_ds["roi_shape_zyx"] == (64, 128, 128)
    assert res_ds["downsampled_shape_zyx"] == (32, 32, 32)
    assert res_ds["peak_ram_mb"] < 35.0
    assert res_ds["fault_step_recovery"] == pytest.approx(1.0, abs=1e-4)
    assert res_ds["beltrami_curl_preservation_ratio"] > res_ds["naive_avg_curl_ratio"] * 1.10

    # Verify CLI `ingest-segy` with --roi, --max-grid, --export-json, and --output-segy
    json_out = tmp_path / "segy_roi_telemetry.json"
    out_segy = tmp_path / "segy_downsampled.segy"
    rc = cli_main(
        [
            "ingest-segy",
            str(segy_path),
            "--roi",
            roi_spec,
            "--max-grid",
            "32",
            "--export-json",
            str(json_out),
            "--output-segy",
            str(out_segy),
        ]
    )
    assert rc == 0
    assert json_out.exists()
    assert out_segy.exists()


def test_z_slab_chunked_bit_identity():
    """
    Evaluate `DarkSnailGeo3DVelocityModeler` and `ContinuousSpectralGeoField3D` on a
    64 x 64 x 64 volume with `chunk_z=8`, `chunk_z=16`, and `chunk_z=64` (unchunked),
    asserting `max_abs_diff < 1e-6`, `0.000e+00` radial drift, and `<= 1.14e-13` SO(6)
    Kelvin-Mandel orthogonality error across all slabs.
    """
    torch.manual_seed(42)
    grid_shape = (64, 64, 64)
    scen = build_geological_scenario("utah_forge_egs_doublet", grid_shape=grid_shape)

    model = DarkSnailGeo3DVelocityModeler(
        grid_shape=grid_shape,
        num_lithologies=len(LITHOLOGY_CATALOG),
        num_fault_modes=len(FAULT_MODE_NAMES),
        canonical_vp_kms=tuple(item["vp_kms"] for item in LITHOLOGY_CATALOG),
    )
    model.configure_fault_throw_rules(scen["fault_pairs"])
    model.eval()

    with torch.no_grad():
        res_unchunked = model(
            ell_sym_3d=scen["ell_sym_3d"],
            u_fault_3d=scen["u_fault_3d"],
            tau_v=0.20,
            triad_gain=1.0,
            D_jacobian=scen["D_jacobian"],
            residual_vp=scen["residual_vp"],
            precision_pi=scen["precision_pi"],
            lambda_ridge=1e-2,
            chunk_z=64,
        )
        res_chunk16 = model(
            ell_sym_3d=scen["ell_sym_3d"],
            u_fault_3d=scen["u_fault_3d"],
            tau_v=0.20,
            triad_gain=1.0,
            D_jacobian=scen["D_jacobian"],
            residual_vp=scen["residual_vp"],
            precision_pi=scen["precision_pi"],
            lambda_ridge=1e-2,
            chunk_z=16,
        )
        res_chunk8 = model(
            ell_sym_3d=scen["ell_sym_3d"],
            u_fault_3d=scen["u_fault_3d"],
            tau_v=0.20,
            triad_gain=1.0,
            D_jacobian=scen["D_jacobian"],
            residual_vp=scen["residual_vp"],
            precision_pi=scen["precision_pi"],
            lambda_ridge=1e-2,
            chunk_z=8,
        )

        spec_unchunked = model.spectral_field(
            strat_field_3d=res_unchunked["vp_cube_kms"],
            obstacle_shell_3d=scen["salt_shell_3d"],
            chunk_z=64,
        )
        spec_chunk16 = model.spectral_field(
            strat_field_3d=res_chunk16["vp_cube_kms"],
            obstacle_shell_3d=scen["salt_shell_3d"],
            chunk_z=16,
        )
        spec_chunk8 = model.spectral_field(
            strat_field_3d=res_chunk8["vp_cube_kms"],
            obstacle_shell_3d=scen["salt_shell_3d"],
            chunk_z=8,
        )

    for key in (
        "vp_cube_kms",
        "vp_tti_cube_kms",
        "vs_cube_kms",
        "rho_cube_gcc",
        "acoustic_impedance_zp",
        "d_ell_triad_3d",
    ):
        diff_16 = float((res_unchunked[key] - res_chunk16[key]).abs().max().item())
        diff_8 = float((res_unchunked[key] - res_chunk8[key]).abs().max().item())
        assert diff_16 < 1e-6, f"chunk_z=16 mismatch on {key}: {diff_16:.3e}"
        assert diff_8 < 1e-6, f"chunk_z=8 mismatch on {key}: {diff_8:.3e}"

    for skey in ("phi_spec_3d", "adelic_gamma_3d", "ray_channels"):
        sdiff_16 = float((spec_unchunked[skey] - spec_chunk16[skey]).abs().max().item())
        sdiff_8 = float((spec_unchunked[skey] - spec_chunk8[skey]).abs().max().item())
        assert sdiff_16 < 1e-6, f"spectral chunk_z=16 mismatch on {skey}: {sdiff_16:.3e}"
        assert sdiff_8 < 1e-6, f"spectral chunk_z=8 mismatch on {skey}: {sdiff_8:.3e}"

    # Verify exact spherical energy orthogonality (0.000e+00 radial drift) & SO(6) Kelvin-Mandel orthogonality
    assert float(res_unchunked["max_orthogonality_residual"]) <= 1e-8
    assert float(res_chunk16["max_orthogonality_residual"]) <= 1e-8
    assert float(res_chunk8["max_orthogonality_residual"]) <= 1e-8
    assert float(res_unchunked["cascade_radial_drift"]) == 0.0
    assert float(res_chunk16["cascade_radial_drift"]) == 0.0
    assert float(res_chunk8["cascade_radial_drift"]) == 0.0
    assert float(res_chunk16["so6_orthogonality_error"]) <= 1.14e-13
    assert float(res_chunk8["so6_orthogonality_error"]) <= 1.14e-13


def test_utah_forge_16a_16b_doublet_spike_recovery():
    """
    Load `data/utah_forge/well_16A_78_32_sonic.las`, `well_16A_78_32.dev`, and `well_16B_78_32.dev`,
    verify total repo footprint < 350 KB, verify `BeltramiWellLogShell` recovers 1.0000 (100.0%)
    spike cosine on the stimulated EGS fracture zones where naive voxel averaging loses >95%
    of the thin-bed slowness spikes, and verify `SkewSymmetricFaultTriad` reduces velocity RMSE
    across the inclined Opal Mound fault by >85% over naive RBF.
    """
    files = ensure_utah_forge_field_files()
    file_paths = [files["dev_16a_path"], files["dev_16b_path"], files["las_16a_path"]]
    total_bytes = sum(p.stat().st_size for p in file_paths)
    assert total_bytes < 350 * 1024, f"Utah FORGE data footprint {total_bytes} B exceeds 350 KB cap"

    traj_16a = read_dev_survey(files["dev_16a_path"], grid_shape=(48, 48, 48))
    traj_16b = read_dev_survey(files["dev_16b_path"], grid_shape=(48, 48, 48))
    las_16a = read_las2_file(files["las_16a_path"], grid_shape=(48, 48, 48))

    # Verify physical geometry of 16A(78)-32 and 16B(78)-32 doublet (300 ft / 91.4 m parallel separation)
    assert traj_16a.md_m[-1] == pytest.approx(3348.8, abs=5.0)
    assert traj_16a.inc_deg[-1] == pytest.approx(65.0, abs=1.0)
    assert traj_16a.azi_deg[-1] == pytest.approx(105.0, abs=1.0)
    assert traj_16b.md_m[-1] == pytest.approx(3336.6, abs=5.0)
    tangent_tvd_diff_m = float(traj_16a.tvd_m[-1] - traj_16b.tvd_m[-1])
    assert 50.0 <= tangent_tvd_diff_m <= 110.0

    # Verify BeltramiWellLogShell on raw 16A sonic log recovers 1.0000 of Stage 1-3 fracture spikes
    # while naive voxel averaging loses >95% of the thin-bed fracture spike contrast
    vp_curve = torch.from_numpy(las_16a.curves["VP"]).float()
    shell = BeltramiWellLogShell(outlier_budget=16, condensed_budget=16)
    shell_res = shell(vp_curve)
    assert float(shell_res.beltrami_spike_recovery) == pytest.approx(1.0, abs=1e-5)

    # Check Stage 1-3 discrete stimulated EGS fracture stringers in granite basement (MD > 1850 m)
    md_arr = las_16a.trajectory.md_m
    vp_arr = las_16a.curves["VP"]
    bel_arr = shell_res.reconstructed_vp.numpy()
    naive_arr = shell_res.naive_pooled_vp.numpy()
    frac_mask = (md_arr > 1850.0) & (vp_arr < 4.20)
    assert int(frac_mask.sum()) >= 3

    # Local intact granite background at each Stage 1-3 depth
    opal_mound_md = 1432.56
    granite_bg = 5.82 + 0.03 * ((md_arr[frac_mask] - opal_mound_md) / (3348.84 - opal_mound_md))
    true_drop = float(np.max(granite_bg - vp_arr[frac_mask]))
    bel_drop = float(np.max(granite_bg - bel_arr[frac_mask]))
    naive_drop = float(np.max(granite_bg - naive_arr[frac_mask]))
    assert bel_drop / true_drop == pytest.approx(1.0, abs=1e-3)
    # Naive voxel pooling across coarse vertical bins dilutes >95% of the sub-meter slowness drop
    assert float(shell_res.naive_spike_recovery) < 0.05
    assert naive_drop / true_drop < 0.05

    # Verify full 3D `utah_forge_egs_doublet` scenario benchmark & Opal Mound fault RMSE reduction (>85%)
    bundle = evaluate_scenario_comparison("utah_forge_egs_doublet", grid_shape=(32, 32, 32))
    forge_cat_names = [u["name"] for u in bundle["scenario"]["lithology_catalog"]]
    assert "Louann Salt Diapir (Halite)" not in forge_cat_names
    assert "Stimulated EGS Hydrothermal Fracture Cloud" in forge_cat_names
    assert "Intact Tertiary Granitoid Basement (16A/16B)" in forge_cat_names
    m = bundle["metrics"]
    assert m["beltrami_spike_recovery"] == pytest.approx(1.0, abs=1e-4)
    assert m["max_orthogonality_residual"] <= 1e-8
    assert m["step0_homotopy_error"] == 0.0

    fault_rmse_reduction = 1.0 - (m["g3vm_fault_zone_rmse_kms"] / m["naive_fault_zone_rmse_kms"])
    assert fault_rmse_reduction > 0.85, (
        f"Expected >85% Opal Mound fault RMSE reduction over naive RBF, got {fault_rmse_reduction * 100:.2f}%"
    )
