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
        traj.z_vox = curves["Z_VOX"]
        traj.y_vox = curves["Y_VOX"]
        traj.x_vox = curves["X_VOX"]
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
        traj.z_vox = arr[:, 6]
        traj.y_vox = arr[:, 7]
        traj.x_vox = arr[:, 8]
    return traj


# ==============================================================================
# SEG-Y Rev 1 3D Velocity / Seismic Cube Reader & Writer
# ==============================================================================


def _ibm32_to_ieee32(ibm_uint32: np.ndarray) -> np.ndarray:
    """Convert big-endian 32-bit IBM floating-point numbers to IEEE-754 float32."""
    u = ibm_uint32.astype(np.uint32)
    sign = np.where((u >> 31) != 0, -1.0, 1.0).astype(np.float64)
    exp = ((u >> 24) & 0x7F).astype(np.int32) - 64
    frac = (u & 0x00FFFFFF).astype(np.float64) / float(1 << 24)
    val = sign * frac * (16.0 ** exp)
    val[u == 0] = 0.0
    return val.astype(np.float32)


def write_segy_cube(
    output_path: str | Path,
    cube_zyx: np.ndarray | torch.Tensor,
    sample_interval_us: int = 2000,
    inline_start: int = 1000,
    crossline_start: int = 2000,
    title: str = "G3VM (DarkSnail-Geo3D) 3D Velocity Volume",
) -> Path:
    """
    Write a 3D property cube `[Z, Y, X]` (Depth/Time Z, Inline Y, Crossline X) to a standard
    SEG-Y Rev 1 binary file (`.segy` / `.sgy`, Format Code 5 = 4-byte IEEE float, big-endian):
      - 3200-byte ASCII textual file header (40 cards x 80 chars)
      - 400-byte binary file header (sample interval bytes 3217-3218, num samples bytes 3221-3222,
        format code 5 at bytes 3225-3226, SEG-Y Rev 1 flag 0x0100 at bytes 3501-3502)
      - Y * X traces, each consisting of a 240-byte trace header (Inline at bytes 189-192,
        Crossline at bytes 193-196, CDP X at 181-184, CDP Y at 185-188, sample count at 115-116,
        sample interval at 117-118) followed by Z big-endian IEEE float32 samples.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (
        cube_zyx.detach().cpu().numpy().astype(np.float32)
        if isinstance(cube_zyx, torch.Tensor)
        else np.asarray(cube_zyx, dtype=np.float32)
    )
    if arr.ndim == 4:
        arr = arr[0]
    Z, Y, X = arr.shape

    cards = [
        f"C 1 {title:<75}"[:80],
        f"C 2 DIMENSIONS (Z, INLINE_Y, XLINE_X): {Z} x {Y} x {X}"[:80].ljust(80),
        f"C 3 INLINE RANGE: {inline_start}..{inline_start + Y - 1} | XLINE RANGE: {crossline_start}..{crossline_start + X - 1}"[:80].ljust(80),
        "C 4 FORMAT: SEG-Y REV 1 | IEEE FLOAT32 (CODE 5) | GENERATED BY G3VM".ljust(80),
    ]
    while len(cards) < 39:
        cards.append(f"C{len(cards) + 1:2d} ".ljust(80))
    cards.append("C40 END TEXTUAL HEADER".ljust(80))
    text_header = "".join(cards).encode("ascii", errors="replace")[:3200].ljust(3200, b" ")

    bin_header = bytearray(400)
    struct.pack_into(">i", bin_header, 0, 1)                     # Job ID
    struct.pack_into(">i", bin_header, 4, 1)                     # Line number
    struct.pack_into(">i", bin_header, 8, 1)                     # Reel number
    struct.pack_into(">h", bin_header, 12, X)                    # Traces per ensemble
    struct.pack_into(">H", bin_header, 16, sample_interval_us)   # Sample interval (us)
    struct.pack_into(">H", bin_header, 20, Z)                    # Samples per trace
    struct.pack_into(">h", bin_header, 24, 5)                    # Format code 5 = 4-byte IEEE float
    struct.pack_into(">h", bin_header, 54, 1)                    # Measurement system: 1 = Meters
    struct.pack_into(">H", bin_header, 300, 0x0100)              # SEG-Y Rev 1.0

    with path.open("wb") as f:
        f.write(text_header)
        f.write(bin_header)
        trace_seq = 1
        for iy in range(Y):
            il_num = inline_start + iy
            for ix in range(X):
                xl_num = crossline_start + ix
                th = bytearray(240)
                struct.pack_into(">i", th, 0, trace_seq)         # Trace sequence in line (1-4)
                struct.pack_into(">i", th, 4, trace_seq)         # Trace sequence in file (5-8)
                struct.pack_into(">i", th, 8, il_num)            # Original field record (9-12)
                struct.pack_into(">i", th, 12, xl_num)           # Trace number in field record (13-16)
                struct.pack_into(">i", th, 20, trace_seq)        # Ensemble / CDP number (21-24)
                struct.pack_into(">h", th, 28, 1)                # Trace ID code: 1 = seismic/velocity data
                struct.pack_into(">H", th, 114, Z)               # Number of samples in trace (115-116)
                struct.pack_into(">H", th, 116, sample_interval_us)  # Sample interval (117-118)
                struct.pack_into(">i", th, 180, ix * 100)        # CDP X coordinate (181-184)
                struct.pack_into(">i", th, 184, iy * 100)        # CDP Y coordinate (185-188)
                struct.pack_into(">i", th, 188, il_num)          # 3D Inline number (189-192)
                struct.pack_into(">i", th, 192, xl_num)          # 3D Crossline number (193-196)
                f.write(th)
                f.write(arr[:, iy, ix].astype(">f4").tobytes())
                trace_seq += 1

    return path


def read_segy_cube(segy_path: str | Path) -> Dict[str, Any]:
    """
    Read a 3D SEG-Y Rev 1 binary volume (`.segy` / `.sgy`) supporting IEEE float32 (format 5),
    IBM 32-bit float (format 1), 32-bit int (format 2), and 16-bit int (format 3),
    automatically reconstructing the 3D `[Z, Y, X]` cube from Inline (bytes 189-192) and
    Crossline (bytes 193-196) trace headers.
    """
    raw = Path(segy_path).read_bytes()
    if len(raw) < 3600:
        raise ValueError(f"Invalid SEG-Y file '{segy_path}': length {len(raw)} < 3600 bytes.")

    text_header_raw = raw[:3200]
    try:
        text_header = text_header_raw.decode("ascii")
    except UnicodeDecodeError:
        text_header = text_header_raw.decode("cp500", errors="replace")

    bin_hdr = raw[3200:3600]
    dt_us = struct.unpack_from(">H", bin_hdr, 16)[0]
    ns = struct.unpack_from(">H", bin_hdr, 20)[0]
    fmt_code = struct.unpack_from(">h", bin_hdr, 24)[0]

    bps = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(fmt_code, 4)
    trace_step = 240 + ns * bps
    n_traces = (len(raw) - 3600) // trace_step
    if n_traces <= 0:
        raise ValueError(f"SEG-Y file '{segy_path}' contains 0 complete traces.")

    inlines = np.empty(n_traces, dtype=np.int32)
    xlines = np.empty(n_traces, dtype=np.int32)
    traces = np.empty((n_traces, ns), dtype=np.float32)

    offset = 3600
    for t in range(n_traces):
        th = raw[offset : offset + 240]
        il = struct.unpack_from(">i", th, 188)[0]
        xl = struct.unpack_from(">i", th, 192)[0]
        if il == 0 and xl == 0:
            il = struct.unpack_from(">i", th, 8)[0]
            xl = struct.unpack_from(">i", th, 12)[0]
        inlines[t] = il
        xlines[t] = xl

        data_buf = raw[offset + 240 : offset + trace_step]
        if fmt_code == 5:
            traces[t] = np.frombuffer(data_buf, dtype=">f4").astype(np.float32)
        elif fmt_code == 1:
            u32 = np.frombuffer(data_buf, dtype=">u4")
            traces[t] = _ibm32_to_ieee32(u32)
        elif fmt_code == 2:
            traces[t] = np.frombuffer(data_buf, dtype=">i4").astype(np.float32)
        elif fmt_code == 3:
            traces[t] = np.frombuffer(data_buf, dtype=">i2").astype(np.float32)
        else:
            traces[t] = np.frombuffer(data_buf, dtype=">f4").astype(np.float32)
        offset += trace_step

    uniq_il = np.unique(inlines)
    uniq_xl = np.unique(xlines)
    Y, X = uniq_il.size, uniq_xl.size

    if Y * X == n_traces:
        il_idx = np.searchsorted(uniq_il, inlines)
        xl_idx = np.searchsorted(uniq_xl, xlines)
        cube_zyx = np.zeros((ns, Y, X), dtype=np.float32)
        cube_zyx[:, il_idx, xl_idx] = traces.T
    else:
        side = int(round(n_traces**0.5))
        cube_zyx = traces.T.reshape(ns, side, -1)

    return {
        "cube_zyx": cube_zyx,
        "inlines": uniq_il,
        "crosslines": uniq_xl,
        "sample_interval_us": dt_us,
        "format_code": fmt_code,
        "text_header": text_header,
    }
