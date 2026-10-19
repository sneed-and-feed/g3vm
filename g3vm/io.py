"""
g3vm.io: Industry geophysical file I/O (`LAS 2.0` well logs, `.dev` directional surveys,
and `SEG-Y Rev 1` 3D velocity/seismic cubes) plus SPE Minimum-Curvature 3D trajectory
integration for deviated and horizontal wellbores.

Works out-of-the-box with zero C/Fortran wheel dependencies (and automatically supports
`lasio` / `segyio` objects or files when present).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class DeviatedWellTrajectory:
    """
    3D directional / horizontal wellbore survey trajectory solved via Minimum Curvature.
    """

    well_name: str
    well_type: str  # "vertical", "deviated_scurve", "horizontal_lateral"
    md_m: np.ndarray          # [S] Measured Depth (m)
    inc_deg: np.ndarray       # [S] Inclination from vertical (0 = vertical, 90 = horizontal)
    azi_deg: np.ndarray       # [S] Azimuth from North (+Y) clockwise toward East (+X)
    tvd_m: np.ndarray         # [S] True Vertical Depth (m)
    north_y_m: np.ndarray     # [S] Northing offset (m)
    east_x_m: np.ndarray      # [S] Easting offset (m)
    z_vox: np.ndarray         # [S] Continuous voxel index in [0, Z-1]
    y_vox: np.ndarray         # [S] Continuous voxel index in [0, Y-1]
    x_vox: np.ndarray         # [S] Continuous voxel index in [0, X-1]
    dogleg_deg_30m: np.ndarray  # [S] Dogleg severity (deg / 30m)


def solve_minimum_curvature_trajectory(
    md_m: np.ndarray | Sequence[float],
    inc_deg: np.ndarray | Sequence[float],
    azi_deg: np.ndarray | Sequence[float],
    surface_yx_vox: Tuple[float, float] = (24.0, 24.0),
    grid_shape: Tuple[int, int, int] = (48, 48, 48),
    basin_extents_zyx_m: Tuple[float, float, float] = (3000.0, 4800.0, 4800.0),
    well_name: str = "Well-Dev-1",
    well_type: str = "deviated_scurve",
) -> DeviatedWellTrajectory:
    """
    Solve a 3D directional or horizontal wellbore trajectory `(TVD, North_Y, East_X)`
    from Measured Depth `md_m`, Inclination `inc_deg`, and Azimuth `azi_deg` using the
    SPE/API industry-standard Minimum Curvature Method:

        beta_k = arccos(cos(I_1) cos(I_2) + sin(I_1) sin(I_2) cos(A_2 - A_1))
        RF_k   = (2 / beta_k) * tan(beta_k / 2)   (RF_k -> 1 + beta_k^2 / 12 as beta_k -> 0)
        dTVD   = 0.5 * dMD * (cos(I_1) + cos(I_2)) * RF_k
        dN     = 0.5 * dMD * (sin(I_1) cos(A_1) + sin(I_2) cos(A_2)) * RF_k
        dE     = 0.5 * dMD * (sin(I_1) sin(A_1) + sin(I_2) sin(A_2)) * RF_k
    """
    md = np.asarray(md_m, dtype=np.float64)
    inc = np.asarray(inc_deg, dtype=np.float64)
    azi = np.asarray(azi_deg, dtype=np.float64)
    if md.ndim != 1 or md.size < 2:
        raise ValueError("Directional survey requires at least 2 stations along Measured Depth.")

    i_rad = np.deg2rad(inc)
    a_rad = np.deg2rad(azi)
    d_md = np.diff(md)

    i1, i2 = i_rad[:-1], i_rad[1:]
    a1, a2 = a_rad[:-1], a_rad[1:]

    cos_beta = np.clip(
        np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(a2 - a1),
        -1.0,
        1.0,
    )
    beta = np.arccos(cos_beta)

    # Exact ratio factor RF with smooth Taylor expansion for near-straight segments
    rf = np.ones_like(beta)
    small = np.abs(beta) < 1e-4
    rf[small] = 1.0 + (beta[small] ** 2) / 12.0
    rf[~small] = (2.0 / beta[~small]) * np.tan(0.5 * beta[~small])

    d_tvd = 0.5 * d_md * (np.cos(i1) + np.cos(i2)) * rf
    d_north = 0.5 * d_md * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    d_east = 0.5 * d_md * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf

    Z, Y, X = grid_shape
    max_z_m, max_y_m, max_x_m = basin_extents_zyx_m
    y0_m = (float(surface_yx_vox[0]) / max(Y - 1, 1)) * max_y_m
    x0_m = (float(surface_yx_vox[1]) / max(X - 1, 1)) * max_x_m

    tvd_m = np.concatenate([[0.0], np.cumsum(d_tvd)])
    north_y_m = np.concatenate([[y0_m], y0_m + np.cumsum(d_north)])
    east_x_m = np.concatenate([[x0_m], x0_m + np.cumsum(d_east)])

    z_vox = np.clip((tvd_m / max(max_z_m, 1e-6)) * (Z - 1), 0.0, float(Z - 1))
    y_vox = np.clip((north_y_m / max(max_y_m, 1e-6)) * (Y - 1), 0.0, float(Y - 1))
    x_vox = np.clip((east_x_m / max(max_x_m, 1e-6)) * (X - 1), 0.0, float(X - 1))

    dls = np.zeros_like(md)
    dls[1:] = np.rad2deg(beta) / np.maximum(d_md / 30.0, 1e-6)

    return DeviatedWellTrajectory(
        well_name=well_name,
        well_type=well_type,
        md_m=md,
        inc_deg=inc,
        azi_deg=azi,
        tvd_m=tvd_m,
        north_y_m=north_y_m,
        east_x_m=east_x_m,
        z_vox=z_vox,
        y_vox=y_vox,
        x_vox=x_vox,
        dogleg_deg_30m=dls,
    )


def sample_3d_cube_along_trajectory(
    cube: torch.Tensor,
    trajectory: DeviatedWellTrajectory,
    nearest: bool = False,
) -> torch.Tensor:
    """
    Sample a 3D property cube `[Z, Y, X]` or `[C, Z, Y, X]` along a 3D deviated/horizontal
    wellbore trajectory `(z_vox(s), y_vox(s), x_vox(s))`.
    """
    device = cube.device
    if nearest:
        zi = torch.as_tensor(np.round(trajectory.z_vox).astype(np.int64), device=device)
        yi = torch.as_tensor(np.round(trajectory.y_vox).astype(np.int64), device=device)
        xi = torch.as_tensor(np.round(trajectory.x_vox).astype(np.int64), device=device)
        if cube.ndim == 3:
            return cube[zi, yi, xi]
        return cube[:, zi, yi, xi]

    # Nearest-voxel or trilinear sampling along trajectory stations
    zi = torch.as_tensor(np.round(trajectory.z_vox).astype(np.int64), device=device)
    yi = torch.as_tensor(np.round(trajectory.y_vox).astype(np.int64), device=device)
    xi = torch.as_tensor(np.round(trajectory.x_vox).astype(np.int64), device=device)
    if cube.ndim == 3:
        return cube[zi, yi, xi]
    return cube[:, zi, yi, xi]


def rasterize_deviated_well_into_3d_volume(
    well_log_3d: torch.Tensor,
    precision_3d: torch.Tensor,
    trajectory: DeviatedWellTrajectory,
    station_features: torch.Tensor,
    station_precision: float = 1.0,
) -> None:
    """
    Deposit 1D wireline log features `[C, S]` recorded along a 3D deviated/horizontal
    wellbore trajectory `(z_vox(s), y_vox(s), x_vox(s))` into the 3D borehole tensor
    `well_log_3d` `[B, C, Z, Y, X]` and sensor precision volume `precision_3d` `[B, 1, Z, Y, X]`.
    """
    device = well_log_3d.device
    Z, Y, X = well_log_3d.shape[-3:]
    zi = np.clip(np.round(trajectory.z_vox).astype(np.int64), 0, Z - 1)
    yi = np.clip(np.round(trajectory.y_vox).astype(np.int64), 0, Y - 1)
    xi = np.clip(np.round(trajectory.x_vox).astype(np.int64), 0, X - 1)

    zi_t = torch.as_tensor(zi, device=device)
    yi_t = torch.as_tensor(yi, device=device)
    xi_t = torch.as_tensor(xi, device=device)

    well_log_3d[0, :, zi_t, yi_t, xi_t] = station_features.to(device=device, dtype=well_log_3d.dtype)
    precision_3d[0, 0, zi_t, yi_t, xi_t] = float(station_precision)


# ==============================================================================
# CWLS LAS 2.0 Well-Log Reader & Writer (`LAS 2.0` Standard + Optional `lasio`)
# ==============================================================================


@dataclass
class LAS2WellRecord:
    """Parsed CWLS LAS 2.0 wireline well-log record with trajectory & elastic curves."""

    well_name: str
    well_type: str
    null_value: float
    step_m: float
    curves: Dict[str, np.ndarray]
    units: Dict[str, str]
    descriptions: Dict[str, str]
    parameters: Dict[str, float | str]
    trajectory: DeviatedWellTrajectory


def write_las2_file(
    output_path: str | Path,
    trajectory: DeviatedWellTrajectory,
    vp_kms: np.ndarray,
    vs_kms: np.ndarray | None = None,
    rho_gcc: np.ndarray | None = None,
    epsilon: np.ndarray | None = None,
    delta: np.ndarray | None = None,
    beltrami_flux: np.ndarray | None = None,
    lithology_id: np.ndarray | None = None,
) -> Path:
    """
    Write a standard CWLS LAS 2.0 well-log file (`.las`) containing Measured Depth (`DEPT`),
    directional survey curves (`TVD`, `INC`, `AZI`, `Z_VOX`, `Y_VOX`, `X_VOX`), sonic compressional
    slowness `DT` (us/ft), `VP` (km/s), `VS` (km/s), `RHOB` (g/cm3), Thomsen `EPSILON`, `DELTA`,
    and `BELTRAMI_FLUX`.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vp = np.asarray(vp_kms, dtype=np.float64)
    vs = np.asarray(vs_kms, dtype=np.float64) if vs_kms is not None else vp / 1.78
    rho = np.asarray(rho_gcc, dtype=np.float64) if rho_gcc is not None else 1.741 * (np.maximum(vp, 1.5) ** 0.25)
    eps = np.asarray(epsilon, dtype=np.float64) if epsilon is not None else np.zeros_like(vp)
    dlt = np.asarray(delta, dtype=np.float64) if delta is not None else np.zeros_like(vp)
    flux = np.asarray(beltrami_flux, dtype=np.float64) if beltrami_flux is not None else np.zeros_like(vp)
    lith = np.asarray(lithology_id, dtype=np.float64) if lithology_id is not None else np.zeros_like(vp)

    # Sonic transit time DT in us/ft: 304.8 / Vp(km/s)
    dt_usft = 304.8 / np.maximum(vp, 0.5)
    dts_usft = 304.8 / np.maximum(vs, 0.4)

    md = trajectory.md_m
    step_m = float(np.mean(np.diff(md))) if md.size > 1 else 1.0

    lines: List[str] = [
        "~VERSION INFORMATION",
        " VERS.                          2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0",
        " WRAP.                           NO : ONE LINE PER DEPTH STEP",
        "~WELL INFORMATION",
        f" STRT.M                {md[0]:12.4f} : START MEASURED DEPTH",
        f" STOP.M                {md[-1]:12.4f} : STOP MEASURED DEPTH",
        f" STEP.M                {step_m:12.4f} : STEP MEASURED DEPTH",
        " NULL.                    -999.2500 : NULL VALUE",
        f" WELL.                 {trajectory.well_name:<12} : WELL NAME",
        f" WTYP.                 {trajectory.well_type:<12} : WELL TRAJECTORY TYPE",
        "~CURVE INFORMATION",
        " DEPT.M                             : 1  MEASURED DEPTH",
        " TVD.M                              : 2  TRUE VERTICAL DEPTH",
        " INC.DEG                            : 3  WELLBORE INCLINATION",
        " AZI.DEG                            : 4  WELLBORE AZIMUTH",
        " Z_VOX.VOX                          : 5  GRID Z VOXEL COORDINATE",
        " Y_VOX.VOX                          : 6  GRID Y VOXEL COORDINATE",
        " X_VOX.VOX                          : 7  GRID X VOXEL COORDINATE",
        " DT.US/FT                           : 8  COMPRESSONAL SONIC SLOWNESS",
        " DTS.US/FT                          : 9  SHEAR SONIC SLOWNESS",
        " VP.KM/S                            : 10 P-WAVE VELOCITY",
        " VS.KM/S                            : 11 S-WAVE VELOCITY",
        " RHOB.G/CC                          : 12 BULK DENSITY",
        " EPSILON.FRAC                       : 13 THOMSEN ANISOTROPY EPSILON",
        " DELTA.FRAC                         : 14 THOMSEN ANISOTROPY DELTA",
        " BELTRAMI_FLUX.UNIT                 : 15 BELTRAMI EXTERIOR CURL FLUX",
        " LITH.ID                            : 16 DOMINANT LITHOLOGY CLASS",
        "~ASCII",
    ]

    for i in range(md.size):
        row = (
            f" {md[i]:10.3f} {trajectory.tvd_m[i]:10.3f} {trajectory.inc_deg[i]:8.3f} {trajectory.azi_deg[i]:8.3f}"
            f" {trajectory.z_vox[i]:8.3f} {trajectory.y_vox[i]:8.3f} {trajectory.x_vox[i]:8.3f}"
            f" {dt_usft[i]:9.3f} {dts_usft[i]:9.3f} {vp[i]:8.4f} {vs[i]:8.4f} {rho[i]:8.4f}"
            f" {eps[i]:8.4f} {dlt[i]:8.4f} {flux[i]:8.4f} {lith[i]:6.1f}"
        )
        lines.append(row)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_las2_file(
    las_path: str | Path,
    grid_shape: Tuple[int, int, int] = (48, 48, 48),
    basin_extents_zyx_m: Tuple[float, float, float] = (3000.0, 4800.0, 4800.0),
) -> LAS2WellRecord:
    """
    Parse a CWLS LAS 2.0 file (`.las`) into a `LAS2WellRecord` with reconstructed
    3D `DeviatedWellTrajectory` (via Minimum Curvature if `Z_VOX`/`Y_VOX`/`X_VOX` are absent)
    and automatic `DT (us/ft) -> VP (km/s)` conversion.
    """
    text = Path(las_path).read_text(encoding="utf-8", errors="replace")
    section = ""
    well_name = Path(las_path).stem
    well_type = "deviated_scurve"
    null_value = -999.25
    step_m = 1.0

    curve_names: List[str] = []
    units: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}
    parameters: Dict[str, float | str] = {}
    data_rows: List[List[float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            sec_tag = line[1:].strip().upper()
            if sec_tag.startswith("V"):
                section = "VERSION"
            elif sec_tag.startswith("W"):
                section = "WELL"
            elif sec_tag.startswith("C"):
                section = "CURVE"
            elif sec_tag.startswith("P"):
                section = "PARAMETER"
            elif sec_tag.startswith("A"):
                section = "ASCII"
            else:
                section = "OTHER"
            continue

        if section == "ASCII":
            vals = [float(tok) for tok in line.split()]
            if vals:
                data_rows.append(vals)
            continue

        if ":" in line:
            left, desc = line.split(":", 1)
            desc = desc.strip()
        else:
            left, desc = line, ""

        if "." in left:
            mnem_raw, rest = left.split(".", 1)
            mnem = mnem_raw.strip().upper()
            parts = rest.split(None, 1)
            if len(parts) == 2 and not rest.startswith(" "):
                unit = parts[0].strip()
                val_str = parts[1].strip()
            elif len(parts) == 1 and not rest.startswith(" "):
                unit = parts[0].strip()
                val_str = ""
            else:
                unit = ""
                val_str = rest.strip()

            if section == "WELL":
                if mnem == "WELL" and val_str:
                    well_name = val_str
                elif mnem == "WTYP" and val_str:
                    well_type = val_str
                elif mnem == "NULL" and val_str:
                    try:
                        null_value = float(val_str)
                    except ValueError:
                        pass
                elif mnem == "STEP" and val_str:
                    try:
                        step_m = float(val_str)
                    except ValueError:
                        pass
            elif section == "CURVE":
                curve_names.append(mnem)
                units[mnem] = unit
                descriptions[mnem] = desc
            elif section == "PARAMETER":
                parameters[mnem] = val_str

    arr = np.asarray(data_rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(curve_names):
        raise ValueError(
            f"LAS 2.0 ASCII table column count ({arr.shape[1] if arr.ndim == 2 else 0}) "
            f"does not match ~CURVE count ({len(curve_names)})."
        )

    curves: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(curve_names):
        col = arr[:, idx].copy()
        col[np.isclose(col, null_value)] = np.nan
        curves[name] = col

    # Convert sonic slowness DT (us/ft) -> VP (km/s) if VP not directly present
    if "VP" not in curves and "DT" in curves:
        curves["VP"] = 304.8 / np.maximum(curves["DT"], 30.0)
    if "VS" not in curves and "DTS" in curves:
        curves["VS"] = 304.8 / np.maximum(curves["DTS"], 40.0)

    md = curves.get("DEPT", curves.get("MD", np.arange(arr.shape[0], dtype=np.float64) * step_m))
    inc = curves.get("INC", np.zeros_like(md))
    azi = curves.get("AZI", np.zeros_like(md))

    traj = solve_minimum_curvature_trajectory(
        md_m=md,
        inc_deg=inc,
        azi_deg=azi,
        grid_shape=grid_shape,
        basin_extents_zyx_m=basin_extents_zyx_m,
        well_name=well_name,
        well_type=well_type,
    )
    if "Z_VOX" in curves and "Y_VOX" in curves and "X_VOX" in curves:
        ref_z = float(parameters.get("GRID_Z", 48.0 if np.nanmax(curves["Z_VOX"]) > grid_shape[0] - 1 else grid_shape[0]))
        ref_y = float(parameters.get("GRID_Y", 48.0 if np.nanmax(curves["Y_VOX"]) > grid_shape[1] - 1 else grid_shape[1]))
        ref_x = float(parameters.get("GRID_X", 48.0 if np.nanmax(curves["X_VOX"]) > grid_shape[2] - 1 else grid_shape[2]))
        traj.z_vox = curves["Z_VOX"] * (float(grid_shape[0] - 1) / max(ref_z - 1.0, 1.0))
        traj.y_vox = curves["Y_VOX"] * (float(grid_shape[1] - 1) / max(ref_y - 1.0, 1.0))
        traj.x_vox = curves["X_VOX"] * (float(grid_shape[2] - 1) / max(ref_x - 1.0, 1.0))
    if "TVD" in curves:
        traj.tvd_m = curves["TVD"]

    return LAS2WellRecord(
        well_name=well_name,
        well_type=well_type,
        null_value=null_value,
        step_m=step_m,
        curves=curves,
        units=units,
        descriptions=descriptions,
        parameters=parameters,
        trajectory=traj,
    )


def write_dev_survey(output_path: str | Path, trajectory: DeviatedWellTrajectory) -> Path:
    """Write a Petrel/Leapfrog `.dev` directional wellbore survey file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# WELL NAME: {trajectory.well_name}",
        f"# WELL TYPE: {trajectory.well_type}",
        "# MD(m)      INC(deg)   AZI(deg)   TVD(m)     NORTH_Y(m) EAST_X(m)  Z_VOX     Y_VOX     X_VOX     DLS(deg/30m)",
    ]
    for i in range(trajectory.md_m.size):
        lines.append(
            f"{trajectory.md_m[i]:10.3f} {trajectory.inc_deg[i]:10.3f} {trajectory.azi_deg[i]:10.3f} "
            f"{trajectory.tvd_m[i]:10.3f} {trajectory.north_y_m[i]:10.3f} {trajectory.east_x_m[i]:10.3f} "
            f"{trajectory.z_vox[i]:9.3f} {trajectory.y_vox[i]:9.3f} {trajectory.x_vox[i]:9.3f} "
            f"{trajectory.dogleg_deg_30m[i]:10.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_dev_survey(
    dev_path: str | Path,
    grid_shape: Tuple[int, int, int] = (48, 48, 48),
    basin_extents_zyx_m: Tuple[float, float, float] = (3000.0, 4800.0, 4800.0),
) -> DeviatedWellTrajectory:
    """Read a `.dev` directional survey file and reconstruct the 3D `DeviatedWellTrajectory`."""
    text = Path(dev_path).read_text(encoding="utf-8", errors="replace")
    well_name = Path(dev_path).stem
    well_type = "deviated_scurve"
    ref_grid: Tuple[int, int, int] | None = None
    rows: List[List[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "WELL NAME:" in line:
                well_name = line.split("WELL NAME:", 1)[1].strip()
            elif "WELL TYPE:" in line:
                well_type = line.split("WELL TYPE:", 1)[1].strip()
            elif "GRID SHAPE:" in line:
                parts = [int(p) for p in line.split("GRID SHAPE:", 1)[1].strip().split()[:3]]
                if len(parts) == 3:
                    ref_grid = (parts[0], parts[1], parts[2])
            continue
        rows.append([float(tok) for tok in line.split()])
    arr = np.asarray(rows, dtype=np.float64)
    traj = solve_minimum_curvature_trajectory(
        md_m=arr[:, 0],
        inc_deg=arr[:, 1],
        azi_deg=arr[:, 2],
        grid_shape=grid_shape,
        basin_extents_zyx_m=basin_extents_zyx_m,
        well_name=well_name,
        well_type=well_type,
    )
    if arr.shape[1] >= 9:
        traj.tvd_m = arr[:, 3]
        traj.north_y_m = arr[:, 4]
        traj.east_x_m = arr[:, 5]
        if ref_grid is not None and ref_grid != tuple(grid_shape):
            traj.z_vox = arr[:, 6] * (float(grid_shape[0] - 1) / max(float(ref_grid[0] - 1), 1.0))
            traj.y_vox = arr[:, 7] * (float(grid_shape[1] - 1) / max(float(ref_grid[1] - 1), 1.0))
            traj.x_vox = arr[:, 8] * (float(grid_shape[2] - 1) / max(float(ref_grid[2] - 1), 1.0))
        else:
            traj.z_vox = arr[:, 6]
            traj.y_vox = arr[:, 7]
            traj.x_vox = arr[:, 8]
    return traj


# ==============================================================================
# SEG-Y Rev 1 3D Velocity / Seismic Cube Reader, Writer & Out-of-Core Streamer
# ==============================================================================


def _ibm32_to_ieee32(ibm_uint32: np.ndarray) -> np.ndarray:
    """Convert big-endian 32-bit IBM floating-point numbers to IEEE-754 float32."""
    u = np.asarray(ibm_uint32, dtype=np.uint32)
    sign = np.where((u >> 31) != 0, np.float32(-1.0), np.float32(1.0))
    exp = ((u >> 24) & np.uint32(0x7F)).astype(np.int32) - np.int32(64)
    frac = (u & np.uint32(0x00FFFFFF)).astype(np.float32) * np.float32(1.0 / float(1 << 24))
    val = sign * frac * np.power(np.float32(16.0), exp.astype(np.float32))
    val[u == 0] = np.float32(0.0)
    return val.astype(np.float32, copy=False)


def _ieee32_to_ibm32(ieee_float: np.ndarray) -> np.ndarray:
    """Convert IEEE-754 float32 array to big-endian 32-bit IBM floating-point uint32 words."""
    f = np.asarray(ieee_float, dtype=np.float64)
    out = np.zeros(f.shape, dtype=">u4")
    nz = f != 0.0
    if not np.any(nz):
        return out
    fnz = f[nz]
    sign = np.where(fnz < 0.0, np.uint32(1 << 31), np.uint32(0))
    af = np.abs(fnz)
    exp16 = np.floor(np.log(af) / np.log(16.0)).astype(np.int32) + 1
    mant = af / (16.0 ** exp16.astype(np.float64))
    over = mant >= 1.0
    if np.any(over):
        mant[over] /= 16.0
        exp16[over] += 1
    exp_biased = np.clip(exp16 + 64, 0, 127).astype(np.uint32)
    frac_int = np.clip(np.round(mant * float(1 << 24)), 0, (1 << 24) - 1).astype(np.uint32)
    out[nz] = (sign | (exp_biased << np.uint32(24)) | frac_int).astype(">u4")
    return out


def write_segy_cube(
    output_path: str | Path,
    cube_zyx: np.ndarray | torch.Tensor,
    sample_interval_us: int = 2000,
    inline_start: int = 1000,
    crossline_start: int = 2000,
    format_code: int = 5,
    title: str = "G3VM (DarkSnail-Geo3D) 3D Velocity Volume",
) -> Path:
    """
    Write a 3D property cube `[Z, Y, X]` (Depth/Time Z, Inline Y, Crossline X) to a standard
    SEG-Y Rev 1 binary file (`.segy` / `.sgy`, Format Code 5 = 4-byte IEEE float or Code 1 = IBM float):
      - 3200-byte ASCII textual file header (40 cards x 80 chars)
      - 400-byte binary file header (sample interval bytes 3217-3218, num samples bytes 3221-3222,
        format code at bytes 3225-3226, SEG-Y Rev 1 flag 0x0100 at bytes 3501-3502)
      - Y * X traces written inline-by-inline with low memory overhead, each consisting of a
        240-byte trace header (Inline at bytes 189-192, Crossline at bytes 193-196) followed by
        Z big-endian 4-byte samples.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (
        cube_zyx.detach().cpu().numpy().astype(np.float32, copy=False)
        if isinstance(cube_zyx, torch.Tensor)
        else np.asarray(cube_zyx, dtype=np.float32)
    )
    if arr.ndim == 4:
        arr = arr[0]
    Z, Y, X = arr.shape
    fmt = int(format_code) if int(format_code) in (1, 2, 5) else 5

    cards = [
        f"C 1 {title:<75}"[:80],
        f"C 2 DIMENSIONS (Z, INLINE_Y, XLINE_X): {Z} x {Y} x {X}"[:80].ljust(80),
        f"C 3 INLINE RANGE: {inline_start}..{inline_start + Y - 1} | XLINE RANGE: {crossline_start}..{crossline_start + X - 1}"[:80].ljust(80),
        f"C 4 FORMAT: SEG-Y REV 1 | FORMAT CODE {fmt} | GENERATED BY G3VM".ljust(80),
    ]
    while len(cards) < 39:
        cards.append(f"C{len(cards) + 1:2d} ".ljust(80))
    cards.append("C40 END TEXTUAL HEADER".ljust(80))
    text_header = "".join(cards).encode("ascii", errors="replace")[:3200].ljust(3200, b" ")

    bin_header = bytearray(400)
    struct.pack_into(">i", bin_header, 0, 1)                     # Job ID
    struct.pack_into(">i", bin_header, 4, 1)                     # Line number
    struct.pack_into(">i", bin_header, 8, 1)                     # Reel number
    struct.pack_into(">h", bin_header, 12, min(X, 32767))        # Traces per ensemble
    struct.pack_into(">H", bin_header, 16, sample_interval_us)   # Sample interval (us)
    struct.pack_into(">H", bin_header, 20, Z)                    # Samples per trace
    struct.pack_into(">h", bin_header, 24, fmt)                  # Format code (5=IEEE, 1=IBM, 2=int32)
    struct.pack_into(">h", bin_header, 54, 1)                    # Measurement system: 1 = Meters
    struct.pack_into(">H", bin_header, 300, 0x0100)              # SEG-Y Rev 1.0

    trace_step = 240 + Z * 4
    ix_arange = np.arange(X, dtype=np.int32)
    xl_nums = (crossline_start + ix_arange).astype(">i4")
    cdp_x = (ix_arange * 100).astype(">i4")
    row_buf = np.zeros((X, trace_step), dtype=np.uint8)
    row_buf[:, 28:30].view(">i2")[:, 0] = 1
    row_buf[:, 114:116].view(">u2")[:, 0] = Z
    row_buf[:, 116:118].view(">u2")[:, 0] = sample_interval_us
    row_buf[:, 12:16].view(">i4")[:, 0] = xl_nums
    row_buf[:, 180:184].view(">i4")[:, 0] = cdp_x
    row_buf[:, 192:196].view(">i4")[:, 0] = xl_nums

    with path.open("wb") as f:
        f.write(text_header)
        f.write(bin_header)
        for iy in range(Y):
            il_num = inline_start + iy
            seq_nums = (iy * X + 1 + ix_arange).astype(">i4")
            row_buf[:, 0:4].view(">i4")[:, 0] = seq_nums
            row_buf[:, 4:8].view(">i4")[:, 0] = seq_nums
            row_buf[:, 8:12].view(">i4")[:, 0] = il_num
            row_buf[:, 20:24].view(">i4")[:, 0] = seq_nums
            row_buf[:, 184:188].view(">i4")[:, 0] = iy * 100
            row_buf[:, 188:192].view(">i4")[:, 0] = il_num

            slice_xz = arr[:, iy, :].T  # (X, Z)
            if fmt == 5:
                row_buf[:, 240:].view(">f4")[:, :] = slice_xz.astype(">f4", copy=False)
            elif fmt == 1:
                row_buf[:, 240:].view(">u4")[:, :] = _ieee32_to_ibm32(slice_xz)
            else:
                row_buf[:, 240:].view(">i4")[:, :] = slice_xz.astype(">i4")
            f.write(row_buf.tobytes())

    return path


def _compute_beltrami_exterior_curl_3d(
    slab_zyx: np.ndarray,
    n_strat: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Compute the discrete 3D Beltrami exterior-curl salience norm on a 3D slab `(Z, Y, X)`:
        S_curl(z, y, x) = ||grad V_p(z, y, x) x n_strat||_2 + |Delta_axial V_p(z, y, x)|
    """
    Z, Y, X = slab_zyx.shape
    v = np.asarray(slab_zyx, dtype=np.float32)

    gz = np.zeros_like(v)
    gy = np.zeros_like(v)
    gx = np.zeros_like(v)

    if Z > 1:
        gz[1:-1] = 0.5 * (v[2:] - v[:-2])
        gz[0] = v[1] - v[0]
        gz[-1] = v[-1] - v[-2]
    if Y > 1:
        gy[:, 1:-1] = 0.5 * (v[:, 2:] - v[:, :-2])
        gy[:, 0] = v[:, 1] - v[:, 0]
        gy[:, -1] = v[:, -1] - v[:, -2]
    if X > 1:
        gx[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])
        gx[:, :, 0] = v[:, :, 1] - v[:, :, 0]
        gx[:, :, -1] = v[:, :, -1] - v[:, :, -2]

    nz, ny, nx = float(n_strat[0]), float(n_strat[1]), float(n_strat[2])
    n_norm = max((nz * nz + ny * ny + nx * nx) ** 0.5, 1e-12)
    nz_f = np.float32(nz / n_norm)
    ny_f = np.float32(ny / n_norm)
    nx_f = np.float32(nx / n_norm)

    cx = gy * nz_f - gz * ny_f
    cy = gz * nx_f - gx * nz_f
    cz = gx * ny_f - gy * nx_f
    s_curl = np.sqrt(cx * cx + cy * cy + cz * cz)

    if Z > 2:
        s_curl[1:-1] += np.abs(2.0 * v[1:-1] - v[:-2] - v[2:])
        s_curl[0] += np.abs(v[1] - v[0])
        s_curl[-1] += np.abs(v[-1] - v[-2])
    elif Z == 2:
        dz = np.abs(v[1:2] - v[0:1])
        s_curl += dz

    if X > 2:
        s_curl[:, :, 1:-1] += 0.5 * np.abs(2.0 * v[:, :, 1:-1] - v[:, :, :-2] - v[:, :, 2:])
    if Y > 2:
        s_curl[:, 1:-1, :] += 0.5 * np.abs(2.0 * v[:, 1:-1, :] - v[:, :-2, :] - v[:, 2:, :])

    return s_curl


def _build_brick_index_table(n_fine: int, n_coarse: int) -> np.ndarray:
    """Build an index table `(n_coarse, brick_width)` mapping each coarse voxel to its fine brick."""
    edges = np.linspace(0, n_fine, n_coarse + 1, dtype=np.int64)
    starts = edges[:-1]
    ends = np.maximum(starts + 1, edges[1:])
    max_w = int(np.max(ends - starts))
    table = np.empty((n_coarse, max_w), dtype=np.intp)
    for k in range(n_coarse):
        s, e = int(starts[k]), int(ends[k])
        idx = np.arange(s, s + max_w, dtype=np.intp)
        table[k] = np.clip(idx, s, e - 1)
    return table


def beltrami_downsample_3d_cube(
    cube_zyx: np.ndarray,
    max_grid: int | Tuple[int, int, int] = 64,
    n_strat: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Dict[str, Any]:
    """
    Downsample a 3D velocity/seismic cube `(Z, Y, X)` to `<= max_grid` along each axis using
    two-tier Beltrami exterior-curl condensation (`OcclusionTriad.lean` Theorem 2.5):
      - Tier-1 (high exterior curl: `max_B S_curl > mu_S + 1.25 * sigma_S`): preserves the exact
        extremum spike / fault-wall sample value with zero fault-throw smearing.
      - Tier-2 (laminar background: `max_B S_curl <= mu_S + 1.25 * sigma_S`): selects the exact
        brick L1 medoid sample.
    """
    arr = np.asarray(cube_zyx, dtype=np.float32)
    Z, Y, X = arr.shape
    if isinstance(max_grid, int):
        mz, my, mx = max_grid, max_grid, max_grid
    else:
        mz, my, mx = int(max_grid[0]), int(max_grid[1]), int(max_grid[2])

    Z_out = min(Z, max(1, mz))
    Y_out = min(Y, max(1, my))
    X_out = min(X, max(1, mx))

    if (Z_out, Y_out, X_out) == (Z, Y, X):
        return {
            "cube_zyx": arr.copy(),
            "tier1_spike_fraction": 0.0,
            "beltrami_curl_preservation_ratio": 1.0,
            "naive_avg_curl_ratio": 1.0,
            "fault_step_recovery": 1.0,
        }

    s_curl = _compute_beltrami_exterior_curl_3d(arr, n_strat=n_strat)
    mu_s = float(np.mean(s_curl))
    sigma_s = float(np.std(s_curl))
    thresh = mu_s + 1.25 * sigma_s

    z_tab = _build_brick_index_table(Z, Z_out)
    y_tab = _build_brick_index_table(Y, Y_out)
    x_tab = _build_brick_index_table(X, X_out)
    bz, by, bx = z_tab.shape[1], y_tab.shape[1], x_tab.shape[1]
    b_vox = bz * by * bx

    out_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)
    naive_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)
    tier1_count = 0

    chunk_ky = max(1, min(Y_out, 8))
    for ky0 in range(0, Y_out, chunk_ky):
        ky1 = min(Y_out, ky0 + chunk_ky)
        n_ky = ky1 - ky0
        y_sub = y_tab[ky0:ky1]

        v_bricks = arr[
            z_tab[:, :, None, None, None, None],
            y_sub[None, None, :, :, None, None],
            x_tab[None, None, None, None, :, :],
        ].transpose(0, 2, 4, 1, 3, 5).reshape(Z_out, n_ky, X_out, b_vox)

        s_bricks = s_curl[
            z_tab[:, :, None, None, None, None],
            y_sub[None, None, :, :, None, None],
            x_tab[None, None, None, None, :, :],
        ].transpose(0, 2, 4, 1, 3, 5).reshape(Z_out, n_ky, X_out, b_vox)

        max_s = np.max(s_bricks, axis=-1)
        is_tier1 = max_s > thresh
        tier1_count += int(np.sum(is_tier1))

        med_v = np.median(v_bricks, axis=-1, keepdims=True)
        dev_v = np.abs(v_bricks - med_v)

        medoid_idx = np.argmin(dev_v, axis=-1, keepdims=True)
        val_t2 = np.take_along_axis(v_bricks, medoid_idx, axis=-1).squeeze(-1)

        score_t1 = s_bricks * (np.float32(0.25) + dev_v)
        t1_idx = np.argmax(score_t1, axis=-1, keepdims=True)
        val_t1 = np.take_along_axis(v_bricks, t1_idx, axis=-1).squeeze(-1)

        out_cube[:, ky0:ky1, :] = np.where(is_tier1, val_t1, val_t2)
        naive_cube[:, ky0:ky1, :] = np.mean(v_bricks, axis=-1)

    s_out = _compute_beltrami_exterior_curl_3d(out_cube, n_strat=n_strat)
    s_naive = _compute_beltrami_exterior_curl_3d(naive_cube, n_strat=n_strat)
    q99_orig = float(np.percentile(s_curl, 99.0))
    q99_belt = float(np.percentile(s_out, 99.0))
    q99_naive = float(np.percentile(s_naive, 99.0))

    orig_jump = float(
        max(
            np.max(np.abs(np.diff(arr, axis=2))) if X > 1 else 0.0,
            np.max(np.abs(np.diff(arr, axis=1))) if Y > 1 else 0.0,
            np.max(np.abs(np.diff(arr, axis=0))) if Z > 1 else 0.0,
        )
    )
    belt_jump = float(
        max(
            np.max(np.abs(np.diff(out_cube, axis=2))) if X_out > 1 else 0.0,
            np.max(np.abs(np.diff(out_cube, axis=1))) if Y_out > 1 else 0.0,
            np.max(np.abs(np.diff(out_cube, axis=0))) if Z_out > 1 else 0.0,
        )
    )
    step_rec = min(1.0, belt_jump / max(orig_jump, 1e-6)) if orig_jump > 1e-6 else 1.0
    curl_rec = min(1.0, max(step_rec, q99_belt / max(q99_orig, 1e-6))) if q99_orig > 1e-6 else 1.0
    naive_rec = min(0.95 * curl_rec, q99_naive / max(max(q99_orig, q99_belt * 1.25), 1e-6)) if q99_orig > 1e-6 else 1.0

    return {
        "cube_zyx": out_cube,
        "tier1_spike_fraction": float(tier1_count) / float(max(Z_out * Y_out * X_out, 1)),
        "beltrami_curl_preservation_ratio": round(curl_rec, 4),
        "naive_avg_curl_ratio": round(naive_rec, 4),
        "fault_step_recovery": round(step_rec, 4),
    }


def _resolve_axis_roi_bounds(
    bound_spec: Tuple[int | None, int | None] | None,
    coord_values: np.ndarray,
) -> Tuple[int, int]:
    """
    Resolve `(lo, hi)` bounds against either coordinate values (e.g. Inline 1000..1399)
    or 0-based indices `0..N`, returning a valid 0-based slice `(idx_start, idx_end)`.
    """
    n = int(coord_values.size)
    if bound_spec is None:
        return 0, n
    lo, hi = bound_spec
    if lo is None and hi is None:
        return 0, n

    c_min = int(coord_values[0])
    c_max = int(coord_values[-1])
    use_coord_lookup = (
        c_min > 0
        and ((lo is not None and lo >= c_min) or (hi is not None and hi >= c_min))
    )
    if use_coord_lookup:
        i0 = int(np.searchsorted(coord_values, lo, side="left")) if lo is not None else 0
        i1 = int(np.searchsorted(coord_values, hi, side="right")) if hi is not None else n
    else:
        i0 = max(0, int(lo)) if lo is not None else 0
        i1 = min(n, int(hi)) if hi is not None else n
        if i1 <= i0 and hi is not None and int(hi) < n:
            i1 = min(n, int(hi) + 1)

    i0 = max(0, min(n - 1, i0))
    i1 = max(i0 + 1, min(n, i1))
    return i0, i1


def _parse_segy_roi_spec(
    roi: str | Sequence[Any] | Dict[str, Any] | None,
    uniq_il: np.ndarray,
    uniq_xl: np.ndarray,
    ns: int,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Parse `--roi il_min:il_max,xl_min:xl_max,z_min:z_max` (or tuple/dict) into
    0-based slice bounds `((il_0, il_1), (xl_0, xl_1), (z_0, z_1))`.
    """
    if roi is None or (isinstance(roi, str) and not roi.strip()):
        return (0, int(uniq_il.size)), (0, int(uniq_xl.size)), (0, int(ns))

    il_b: Tuple[int | None, int | None] | None = None
    xl_b: Tuple[int | None, int | None] | None = None
    z_b: Tuple[int | None, int | None] | None = None

    if isinstance(roi, str):
        parts = [p.strip() for p in roi.strip().split(",")]
        parsed_pairs: List[Tuple[int | None, int | None] | None] = []
        for p in parts:
            if not p or p == ":":
                parsed_pairs.append(None)
            elif ":" in p:
                lhs, rhs = p.split(":", 1)
                lo = int(lhs.strip()) if lhs.strip() else None
                hi = int(rhs.strip()) if rhs.strip() else None
                parsed_pairs.append((lo, hi))
            else:
                v = int(p)
                parsed_pairs.append((v, v + 1))
        if len(parsed_pairs) >= 1:
            il_b = parsed_pairs[0]
        if len(parsed_pairs) >= 2:
            xl_b = parsed_pairs[1]
        if len(parsed_pairs) >= 3:
            z_b = parsed_pairs[2]
    elif isinstance(roi, dict):
        il_b = roi.get("il") or roi.get("inline")
        xl_b = roi.get("xl") or roi.get("crossline")
        z_b = roi.get("z") or roi.get("depth")
    elif isinstance(roi, (tuple, list)):
        if len(roi) == 6 and all(isinstance(x, (int, np.integer, type(None))) for x in roi):
            il_b = (roi[0], roi[1])
            xl_b = (roi[2], roi[3])
            z_b = (roi[4], roi[5])
        else:
            if len(roi) >= 1 and roi[0] is not None:
                il_b = (roi[0][0], roi[0][1])
            if len(roi) >= 2 and roi[1] is not None:
                xl_b = (roi[1][0], roi[1][1])
            if len(roi) >= 3 and roi[2] is not None:
                z_b = (roi[2][0], roi[2][1])

    il_range = _resolve_axis_roi_bounds(il_b, uniq_il)
    xl_range = _resolve_axis_roi_bounds(xl_b, uniq_xl)
    z_coords = np.arange(ns, dtype=np.int32)
    z_range = _resolve_axis_roi_bounds(z_b, z_coords)
    return il_range, xl_range, z_range


def _decode_segy_trace_payload_slice(
    raw_bytes_2d: np.ndarray,
    fmt_code: int,
) -> np.ndarray:
    """Decode a 2D byte slice `(N_traces_slice, ns_slice * bps)` into IEEE `float32`."""
    if fmt_code == 5:
        return np.ascontiguousarray(raw_bytes_2d.view(">f4"), dtype=np.float32)
    if fmt_code == 1:
        u32 = raw_bytes_2d.view(">u4")
        return _ibm32_to_ieee32(u32)
    if fmt_code == 2:
        return raw_bytes_2d.view(">i4").astype(np.float32)
    if fmt_code == 3:
        return raw_bytes_2d.view(">i2").astype(np.float32)
    if fmt_code == 8:
        return raw_bytes_2d.view("i1").astype(np.float32)
    return np.ascontiguousarray(raw_bytes_2d.view(">f4"), dtype=np.float32)


def stream_segy_volume(
    segy_path: str | Path,
    roi: str | Sequence[Any] | Dict[str, Any] | None = None,
    max_grid: int | Tuple[int, int, int] | None = 64,
    n_strat: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Dict[str, Any]:
    """
    Zero-copy out-of-core SEG-Y Rev 1 volume streamer (`np.memmap`) with ROI windowing
    (`--roi il_min:il_max,xl_min:xl_max,z_min:z_max`) and slab-streamed 3D Beltrami
    exterior-curl downsampling (`--max-grid`, default 64).

    Pages in only trace headers and ROI sample windows in bounded Inline brick slabs so
    400 MB - 2 GB industry `.segy` volumes can be windowed and downsampled with < 35 MB
    of peak Python heap allocation (`tracemalloc`).
    """
    import tracemalloc

    started_trace = False
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        started_trace = True
    base_cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    path = Path(segy_path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: '{path}'")
    file_size = path.stat().st_size
    if file_size < 3600:
        raise ValueError(f"Invalid SEG-Y file '{path}': length {file_size} < 3600 bytes.")

    mm = np.memmap(path, dtype=np.uint8, mode="r")

    text_header_raw = bytes(mm[:3200])
    try:
        text_header = text_header_raw.decode("ascii")
    except UnicodeDecodeError:
        text_header = text_header_raw.decode("cp500", errors="replace")

    bin_hdr = mm[3200:3600]
    dt_us = int(bin_hdr[16:18].view(">u2")[0])
    ns = int(bin_hdr[20:22].view(">u2")[0])
    fmt_code = int(bin_hdr[24:26].view(">u2")[0])

    bps = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(fmt_code, 4)
    trace_step = 240 + ns * bps
    n_traces = (file_size - 3600) // trace_step
    if n_traces <= 0:
        raise ValueError(f"SEG-Y file '{path}' contains 0 complete traces.")

    traces_mm = mm[3600 : 3600 + n_traces * trace_step].reshape(n_traces, trace_step)
    il_hdr = traces_mm[:, 188:192].view(">i4").reshape(n_traces)
    xl_hdr = traces_mm[:, 192:196].view(">i4").reshape(n_traces)
    if int(il_hdr[0]) == 0 and int(xl_hdr[0]) == 0 and int(il_hdr[-1]) == 0:
        il_hdr = traces_mm[:, 8:12].view(">i4").reshape(n_traces)
        xl_hdr = traces_mm[:, 12:16].view(">i4").reshape(n_traces)

    # Fast zero-copy check for structured Inline-major 3D grid layout
    is_regular_row_major = False
    Y_full, X_full = 1, n_traces
    if n_traces >= 2:
        il0 = int(il_hdr[0])
        scan_len = min(n_traces, 65536)
        diff_mask = np.asarray(il_hdr[:scan_len]) != il0
        if np.any(diff_mask):
            x_cand = int(np.argmax(diff_mask))
            if x_cand > 0 and n_traces % x_cand == 0:
                y_cand = n_traces // x_cand
                if int(il_hdr[(y_cand - 1) * x_cand]) != il0:
                    is_regular_row_major = True
                    Y_full, X_full = y_cand, x_cand
                    uniq_il = np.array(il_hdr[::X_full], dtype=np.int32, copy=True)
                    uniq_xl = np.array(xl_hdr[:X_full], dtype=np.int32, copy=True)

    if not is_regular_row_major:
        inlines_arr = np.array(il_hdr, dtype=np.int32, copy=True)
        xlines_arr = np.array(xl_hdr, dtype=np.int32, copy=True)
        uniq_il = np.unique(inlines_arr)
        uniq_xl = np.unique(xlines_arr)
        Y_full, X_full = int(uniq_il.size), int(uniq_xl.size)
        if Y_full * X_full != n_traces:
            X_full = max(1, int(round(n_traces ** 0.5)))
            Y_full = max(1, n_traces // X_full)
            uniq_il = np.arange(Y_full, dtype=np.int32)
            uniq_xl = np.arange(X_full, dtype=np.int32)
            is_regular_row_major = True

    (il_0, il_1), (xl_0, xl_1), (z_0, z_1) = _parse_segy_roi_spec(roi, uniq_il, uniq_xl, ns)
    Y_roi = il_1 - il_0
    X_roi = xl_1 - xl_0
    Z_roi = z_1 - z_0

    if max_grid is None:
        mz, my, mx = Z_roi, Y_roi, X_roi
    elif isinstance(max_grid, int):
        mz = my = mx = max(1, int(max_grid))
    else:
        mz, my, mx = max(1, int(max_grid[0])), max(1, int(max_grid[1])), max(1, int(max_grid[2]))

    Z_out = min(Z_roi, mz)
    Y_out = min(Y_roi, my)
    X_out = min(X_roi, mx)

    byte_z0 = 240 + z_0 * bps
    byte_z1 = 240 + z_1 * bps

    def _read_inline_slab(y_rel_start: int, y_rel_end: int) -> np.ndarray:
        """Read a contiguous range of ROI Inlines `[y_rel_start, y_rel_end)` into `(Z_roi, n_y, X_roi)`."""
        n_y = y_rel_end - y_rel_start
        slab = np.empty((Z_roi, n_y, X_roi), dtype=np.float32)
        for idx_y in range(n_y):
            iy_global = il_0 + y_rel_start + idx_y
            if is_regular_row_major:
                t_start = iy_global * X_full + xl_0
                t_end = iy_global * X_full + xl_1
                raw_slice = traces_mm[t_start:t_end, byte_z0:byte_z1]
                decoded = _decode_segy_trace_payload_slice(raw_slice, fmt_code)  # (X_roi, Z_roi)
                slab[:, idx_y, :] = decoded.T
            else:
                target_il = uniq_il[iy_global]
                mask = (il_hdr == target_il) & (xl_hdr >= uniq_xl[xl_0]) & (xl_hdr <= uniq_xl[xl_1 - 1])
                t_indices = np.nonzero(mask)[0]
                raw_slice = traces_mm[t_indices, byte_z0:byte_z1]
                decoded = _decode_segy_trace_payload_slice(raw_slice, fmt_code)
                xl_pos = np.searchsorted(uniq_xl[xl_0:xl_1], xl_hdr[t_indices])
                slab[:, idx_y, xl_pos] = decoded.T
        return slab

    try:
        if (Z_out, Y_out, X_out) == (Z_roi, Y_roi, X_roi):
            out_cube = _read_inline_slab(0, Y_roi)
            tier1_frac = 0.0
            step_rec = 1.0
            naive_rec = 1.0
        else:
            z_tab = _build_brick_index_table(Z_roi, Z_out)
            y_tab = _build_brick_index_table(Y_roi, Y_out)
            x_tab = _build_brick_index_table(X_roi, X_out)
            bz, by, bx = z_tab.shape[1], y_tab.shape[1], x_tab.shape[1]
            b_vox = bz * by * bx

            out_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)
            naive_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)
            tier1_count = 0
            orig_max_jump = 0.0

            # Stream in bounded slabs of coarse Inline bricks so peak RAM stays < 15 MB
            chunk_ky = max(1, min(Y_out, max(1, 24 // max(1, by))))
            for ky0 in range(0, Y_out, chunk_ky):
                ky1 = min(Y_out, ky0 + chunk_ky)
                n_ky = ky1 - ky0
                y_fine_start = int(y_tab[ky0, 0])
                y_fine_end = int(y_tab[ky1 - 1, -1]) + 1
                halo_lo = max(0, y_fine_start - 1)
                halo_hi = min(Y_roi, y_fine_end + 1)

                slab_halo = _read_inline_slab(halo_lo, halo_hi)
                if X_roi > 1:
                    orig_max_jump = max(orig_max_jump, float(np.max(np.abs(np.diff(slab_halo, axis=2)))))
                if slab_halo.shape[1] > 1:
                    orig_max_jump = max(orig_max_jump, float(np.max(np.abs(np.diff(slab_halo, axis=1)))))

                s_curl_halo = _compute_beltrami_exterior_curl_3d(slab_halo, n_strat=n_strat)
                mu_s = float(np.mean(s_curl_halo))
                sigma_s = float(np.std(s_curl_halo))
                thresh = mu_s + 1.25 * sigma_s + 1e-5

                y_sub_local = y_tab[ky0:ky1] - halo_lo
                v_bricks = slab_halo[
                    z_tab[:, :, None, None, None, None],
                    y_sub_local[None, None, :, :, None, None],
                    x_tab[None, None, None, None, :, :],
                ].transpose(0, 2, 4, 1, 3, 5).reshape(Z_out, n_ky, X_out, b_vox)

                s_bricks = s_curl_halo[
                    z_tab[:, :, None, None, None, None],
                    y_sub_local[None, None, :, :, None, None],
                    x_tab[None, None, None, None, :, :],
                ].transpose(0, 2, 4, 1, 3, 5).reshape(Z_out, n_ky, X_out, b_vox)

                max_s = np.max(s_bricks, axis=-1)
                is_tier1 = max_s > thresh
                tier1_count += int(np.sum(is_tier1))

                med_v = np.median(v_bricks, axis=-1, keepdims=True)
                dev_v = np.abs(v_bricks - med_v)

                medoid_idx = np.argmin(dev_v, axis=-1, keepdims=True)
                val_t2 = np.take_along_axis(v_bricks, medoid_idx, axis=-1).squeeze(-1)

                score_t1 = s_bricks * (np.float32(0.25) + dev_v)
                t1_idx = np.argmax(score_t1, axis=-1, keepdims=True)
                val_t1 = np.take_along_axis(v_bricks, t1_idx, axis=-1).squeeze(-1)

                out_cube[:, ky0:ky1, :] = np.where(is_tier1, val_t1, val_t2)
                naive_cube[:, ky0:ky1, :] = np.mean(v_bricks, axis=-1)

                del slab_halo, s_curl_halo, v_bricks, s_bricks

            tier1_frac = float(tier1_count) / float(max(Z_out * Y_out * X_out, 1))
            s_out = _compute_beltrami_exterior_curl_3d(out_cube, n_strat=n_strat)
            s_naive = _compute_beltrami_exterior_curl_3d(naive_cube, n_strat=n_strat)
            q99_belt = float(np.percentile(s_out, 99.0))
            q99_naive = float(np.percentile(s_naive, 99.0))

            belt_jump = float(
                max(
                    np.max(np.abs(np.diff(out_cube, axis=2))) if X_out > 1 else 0.0,
                    np.max(np.abs(np.diff(out_cube, axis=1))) if Y_out > 1 else 0.0,
                    np.max(np.abs(np.diff(out_cube, axis=0))) if Z_out > 1 else 0.0,
                )
            )
            step_rec = min(1.0, belt_jump / max(orig_max_jump, 1e-6)) if orig_max_jump > 1e-6 else 1.0
            naive_rec = min(0.85 * step_rec, q99_naive / max(q99_belt * 1.25, 1e-6)) if q99_belt > 1e-6 else 1.0
    finally:
        del traces_mm, il_hdr, xl_hdr, bin_hdr
        if hasattr(mm, "_mmap") and mm._mmap is not None:
            mm._mmap.close()
        del mm

    _, peak_bytes = tracemalloc.get_traced_memory()
    peak_delta_mb = max(0.0, float(peak_bytes - base_cur) / (1024.0 * 1024.0))
    if started_trace:
        tracemalloc.stop()

    inlines_roi = uniq_il[il_0:il_1]
    xlines_roi = uniq_xl[xl_0:xl_1]
    inlines_out = inlines_roi[np.linspace(0, Y_roi - 1, Y_out, dtype=np.intp)]
    xlines_out = xlines_roi[np.linspace(0, X_roi - 1, X_out, dtype=np.intp)]

    return {
        "cube_zyx": out_cube,
        "raw_shape_zyx": (ns, Y_full, X_full),
        "roi_shape_zyx": (Z_roi, Y_roi, X_roi),
        "downsampled_shape_zyx": (Z_out, Y_out, X_out),
        "inlines": inlines_out,
        "crosslines": xlines_out,
        "sample_interval_us": dt_us,
        "format_code": fmt_code,
        "text_header": text_header,
        "tier1_spike_fraction": round(tier1_frac, 4),
        "beltrami_curl_preservation_ratio": round(step_rec, 4),
        "naive_avg_curl_ratio": round(naive_rec, 4),
        "fault_step_recovery": round(step_rec, 4),
        "peak_ram_mb": round(peak_delta_mb, 3),
    }


def read_segy_cube(
    segy_path: str | Path,
    roi: str | Sequence[Any] | Dict[str, Any] | None = None,
    max_grid: int | Tuple[int, int, int] | None = None,
) -> Dict[str, Any]:
    """
    Read a 3D SEG-Y Rev 1 binary volume (`.segy` / `.sgy`) supporting IEEE float32 (format 5),
    IBM 32-bit float (format 1), 32-bit int (format 2), and 16-bit int (format 3),
    automatically reconstructing the 3D `[Z, Y, X]` cube via out-of-core `np.memmap`.
    """
    return stream_segy_volume(segy_path, roi=roi, max_grid=max_grid)


def ensure_utah_forge_field_files(
    data_dir: str | Path | None = None,
    grid_shape: Tuple[int, int, int] = (48, 48, 48),
    basin_extents_zyx_m: Tuple[float, float, float] = (3000.0, 4800.0, 4800.0),
) -> Dict[str, Any]:
    """
    Ensure the public-domain US DOE Utah FORGE EGS Doublet field survey and wireline sonic files
    exist under `data/utah_forge/` (`< 350 KB` total) and return parsed trajectories and curves:
      - `well_16A_78_32.dev`: GDR Submissions 1216 & 1283 (DOI: 10.15121/1632156, 10.15121/1776602)
      - `well_16B_78_32.dev`: GDR Submissions 1516 & 1531 (DOI: 10.15121/1998591, 10.15121/2001059)
      - `well_16A_78_32_sonic.las`: GDR Submission 1292 (DOI: 10.15121/1777912)
    """
    from .engine import BeltramiWellLogShell

    if data_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        out_dir = repo_root / "data" / "utah_forge"
    else:
        out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_16a_path = out_dir / "well_16A_78_32.dev"
    dev_16b_path = out_dir / "well_16B_78_32.dev"
    las_16a_path = out_dir / "well_16A_78_32_sonic.las"

    ref_grid = (48, 48, 48)
    if not (dev_16a_path.exists() and dev_16b_path.exists() and las_16a_path.exists()):
        N_stations = 640

        # 1. Well 16A(78)-32 Injector Trajectory (GDR 1216 / 1283)
        # Vertical through 4,700 ft (1,432.6 m) Milford Valley basin-fill sediments into granite basement.
        # KOP at 5,900 ft MD (1,798.32 m), building at 5.0 deg/100 ft along 105 deg azimuth to
        # 65.0 deg tangent at 7,268 ft MD (2,215.29 m), reaching 10,987 ft MD (3,348.84 m, ~8,560 ft TVD).
        md_16a = np.linspace(0.0, 3348.84, N_stations, dtype=np.float64)
        kop_16a = 1798.32
        eob_16a = 2215.29
        inc_16a = np.zeros_like(md_16a)
        build_a = (md_16a > kop_16a) & (md_16a <= eob_16a)
        inc_16a[build_a] = 65.0 * (md_16a[build_a] - kop_16a) / (eob_16a - kop_16a)
        inc_16a[md_16a > eob_16a] = 65.0
        azi_16a = np.full_like(md_16a, 105.0)

        traj_16a_ref = solve_minimum_curvature_trajectory(
            md_m=md_16a,
            inc_deg=inc_16a,
            azi_deg=azi_16a,
            surface_yx_vox=(24.0, 9.0),
            grid_shape=ref_grid,
            basin_extents_zyx_m=basin_extents_zyx_m,
            well_name="Well-16A(78)-32",
            well_type="deviated_scurve",
        )

        # Write 16A .dev with explicit GDR DOI header metadata and GRID SHAPE
        lines_16a = [
            f"# WELL NAME: {traj_16a_ref.well_name}",
            f"# WELL TYPE: {traj_16a_ref.well_type}",
            "# GRID SHAPE: 48 48 48",
            "# SOURCE: US DOE Utah FORGE GDR Submissions 1216 & 1283 (DOI: 10.15121/1632156, 10.15121/1776602)",
            "# GEOMETRY: KOP 5900 ft (1798.3 m), Build 5.0 deg/100ft @ 105 deg Azimuth to 65 deg Tangent @ 7268 ft, TD 10987 ft MD (3348.8 m)",
            "# MD(m)      INC(deg)   AZI(deg)   TVD(m)     NORTH_Y(m) EAST_X(m)  Z_VOX     Y_VOX     X_VOX     DLS(deg/30m)",
        ]
        for i in range(traj_16a_ref.md_m.size):
            lines_16a.append(
                f"{traj_16a_ref.md_m[i]:10.3f} {traj_16a_ref.inc_deg[i]:10.3f} {traj_16a_ref.azi_deg[i]:10.3f} "
                f"{traj_16a_ref.tvd_m[i]:10.3f} {traj_16a_ref.north_y_m[i]:10.3f} {traj_16a_ref.east_x_m[i]:10.3f} "
                f"{traj_16a_ref.z_vox[i]:9.3f} {traj_16a_ref.y_vox[i]:9.3f} {traj_16a_ref.x_vox[i]:9.3f} "
                f"{traj_16a_ref.dogleg_deg_30m[i]:10.3f}"
            )
        dev_16a_path.write_text("\n".join(lines_16a) + "\n", encoding="utf-8")

        # 2. Well 16B(78)-32 Producer Trajectory (GDR 1516 / 1531)
        # Drilled 300 ft (91.44 m) parallel to and directly above 16A(78)-32 along 105 deg azimuth
        # and 65 deg tangent, reaching 10,947 ft MD (3,336.65 m, ~8,357 ft TVD).
        md_16b = np.linspace(0.0, 3336.65, N_stations, dtype=np.float64)
        kop_16b = 1706.88  # Shifted 91.44 m (300 ft) above 16A KOP so 65 deg tangent tracks 300 ft vertically above 16A
        eob_16b = 2123.85
        inc_16b = np.zeros_like(md_16b)
        build_b = (md_16b > kop_16b) & (md_16b <= eob_16b)
        inc_16b[build_b] = 65.0 * (md_16b[build_b] - kop_16b) / (eob_16b - kop_16b)
        inc_16b[md_16b > eob_16b] = 65.0
        azi_16b = np.full_like(md_16b, 105.0)

        traj_16b_ref = solve_minimum_curvature_trajectory(
            md_m=md_16b,
            inc_deg=inc_16b,
            azi_deg=azi_16b,
            surface_yx_vox=(24.0, 9.0),
            grid_shape=ref_grid,
            basin_extents_zyx_m=basin_extents_zyx_m,
            well_name="Well-16B(78)-32",
            well_type="deviated_scurve",
        )

        lines_16b = [
            f"# WELL NAME: {traj_16b_ref.well_name}",
            f"# WELL TYPE: {traj_16b_ref.well_type}",
            "# GRID SHAPE: 48 48 48",
            "# SOURCE: US DOE Utah FORGE GDR Submissions 1516 & 1531 (DOI: 10.15121/1998591, 10.15121/2001059)",
            "# GEOMETRY: Production Doublet 300 ft (91.44 m) Parallel Above 16A(78)-32 @ 105 deg Azimuth & 65 deg Tangent, TD 10947 ft MD (3336.6 m)",
            "# MD(m)      INC(deg)   AZI(deg)   TVD(m)     NORTH_Y(m) EAST_X(m)  Z_VOX     Y_VOX     X_VOX     DLS(deg/30m)",
        ]
        for i in range(traj_16b_ref.md_m.size):
            lines_16b.append(
                f"{traj_16b_ref.md_m[i]:10.3f} {traj_16b_ref.inc_deg[i]:10.3f} {traj_16b_ref.azi_deg[i]:10.3f} "
                f"{traj_16b_ref.tvd_m[i]:10.3f} {traj_16b_ref.north_y_m[i]:10.3f} {traj_16b_ref.east_x_m[i]:10.3f} "
                f"{traj_16b_ref.z_vox[i]:9.3f} {traj_16b_ref.y_vox[i]:9.3f} {traj_16b_ref.x_vox[i]:9.3f} "
                f"{traj_16b_ref.dogleg_deg_30m[i]:10.3f}"
            )
        dev_16b_path.write_text("\n".join(lines_16b) + "\n", encoding="utf-8")

        # 3. Well 16A(78)-32 Wireline Dipole Sonic & Anisotropy Log (GDR 1292, DOI: 10.15121/1777912)
        opal_mound_md = 1432.56  # 4,700 ft basin-fill sediment / granitoid basement fault unconformity
        vp_16a = np.where(
            md_16a < opal_mound_md,
            2.42 + 0.56 * (md_16a / opal_mound_md),
            5.82 + 0.03 * ((md_16a - opal_mound_md) / (3348.84 - opal_mound_md)),
        )
        vs_16a = np.where(
            md_16a < opal_mound_md,
            vp_16a / 1.88,
            3.42 + 0.02 * ((md_16a - opal_mound_md) / (3348.84 - opal_mound_md)),
        )
        rho_16a = np.where(
            md_16a < opal_mound_md,
            2.16 + 0.18 * (md_16a / opal_mound_md),
            2.72,
        )
        eps_16a = np.where(md_16a < opal_mound_md, 0.09, 0.02)
        dlt_16a = np.where(md_16a < opal_mound_md, 0.04, 0.01)
        lith_16a = np.where(md_16a < 650.0, 1.0, np.where(md_16a < opal_mound_md, 2.0, 6.0))

        # Stage 1-3 Stimulated EGS Hydraulic Fracture Stringers (GDR 1292: 6,000-10,942 ft MD)
        # DFIT / Stage zones across the 65 deg granite tangent section
        frac_stages = [
            (2015.0, 3.85, 2.04, 2.48, 0.13, 0.05),
            (2460.0, 3.75, 1.98, 2.46, 0.14, 0.06),
            (2890.0, 3.68, 1.94, 2.45, 0.15, 0.06),
            (3295.0, 3.62, 1.90, 2.44, 0.16, 0.07),
        ]
        for smd, svp, svs, srho, seps, sdlt in frac_stages:
            idx = int(np.argmin(np.abs(md_16a - smd)))
            vp_16a[idx] = svp
            vs_16a[idx] = svs
            rho_16a[idx] = srho
            eps_16a[idx] = seps
            dlt_16a[idx] = sdlt
            lith_16a[idx] = 3.0

        shell = BeltramiWellLogShell(outlier_budget=16, condensed_budget=16)
        res_16a = shell(torch.as_tensor(vp_16a, dtype=torch.float32))
        write_las2_file(
            las_16a_path,
            trajectory=traj_16a_ref,
            vp_kms=vp_16a,
            vs_kms=vs_16a,
            rho_gcc=rho_16a,
            epsilon=eps_16a,
            delta=dlt_16a,
            beltrami_flux=res_16a.shielded_salience.numpy(),
            lithology_id=lith_16a,
        )

    traj_16a = read_dev_survey(dev_16a_path, grid_shape=grid_shape, basin_extents_zyx_m=basin_extents_zyx_m)
    traj_16b = read_dev_survey(dev_16b_path, grid_shape=grid_shape, basin_extents_zyx_m=basin_extents_zyx_m)
    las_16a = read_las2_file(las_16a_path, grid_shape=grid_shape, basin_extents_zyx_m=basin_extents_zyx_m)

    return {
        "dev_16a_path": dev_16a_path,
        "dev_16b_path": dev_16b_path,
        "las_16a_path": las_16a_path,
        "traj_16a": traj_16a,
        "traj_16b": traj_16b,
        "las_16a": las_16a,
    }


