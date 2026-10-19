"""
g3vm.anisotropy: Tilted Transverse Isotropy (Thomsen VTI/TTI parameters epsilon, delta, gamma),
coupled elastic cubes (Vp, Vs, density rho, acoustic impedance Zp, Vp/Vs, Poisson's ratio),
and Lie-algebraic so(6) Kelvin-Mandel Voigt stiffness tensor rotors across folded strata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 8-lithology coupled elastic & Thomsen VTI/TTI catalog:
# (name, vp0_kms, vs0_kms, rho_gcc, epsilon, delta, gamma)
ELASTIC_TTI_CATALOG: Tuple[Tuple[str, float, float, float, float, float, float], ...] = (
    ("unconsolidated_sand", 1.85, 0.98, 2.05, 0.03, 0.01, 0.02),
    ("overpressured_shale", 2.35, 1.18, 2.24, 0.22, 0.09, 0.20),
    ("interbedded_silt", 2.95, 1.58, 2.36, 0.15, 0.06, 0.13),
    ("reservoir_sandstone", 3.65, 2.18, 2.44, 0.05, 0.02, 0.04),
    ("tight_limestone", 4.75, 2.68, 2.66, 0.07, 0.03, 0.05),
    ("dolomite_stringer", 5.45, 3.12, 2.81, 0.04, 0.02, 0.03),
    ("anhydrite_evaporite", 5.85, 3.28, 2.94, 0.02, 0.01, 0.02),
    ("halite_salt_diapir", 4.50, 2.60, 2.16, 0.00, 0.00, 0.00),
)


@dataclass
class ElasticTTICubeBundle:
    """Complete 3D anisotropic elastic property volume bundle."""

    vp0_kms: torch.Tensor
    vs0_kms: torch.Tensor
    rho_gcc: torch.Tensor
    epsilon: torch.Tensor
    delta: torch.Tensor
    gamma: torch.Tensor
    dip_deg: torch.Tensor
    azimuth_deg: torch.Tensor
    vp_tti_kms: torch.Tensor
    vsv_tti_kms: torch.Tensor
    vsh_tti_kms: torch.Tensor
    vp_vs_ratio: torch.Tensor
    poisson_ratio: torch.Tensor
    acoustic_impedance_zp: torch.Tensor
    stiffness_trace_drift: float
    stiffness_frobenius_drift: float
    so6_orthogonality_error: float = 0.0


def build_kelvin_mandel_so6_rotor(rotation_so3: torch.Tensor) -> torch.Tensor:
    """
    Lift a batch of 3D bedding-plane rotation matrices R in SO(3) [..., 3, 3]
    to the exact isometric 6x6 Kelvin-Mandel orthogonal stiffness operator M(R) in SO(6) [..., 6, 6].

    In the normalized Kelvin-Mandel basis:
        (sigma_11, sigma_22, sigma_33, sqrt(2)*sigma_23, sqrt(2)*sigma_13, sqrt(2)*sigma_12)
    the lifted operator M(R) satisfies M(R)^T M(R) = I_6 (Theorem 2.7: so6_kelvin_stiffness_isometry),
    guaranteeing zero spectral/Frobenius norm drift when rotating VTI stiffness tensors into TTI strata.
    """
    orig_dtype = rotation_so3.dtype
    R = rotation_so3.to(torch.float64)
    r11, r12, r13 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r21, r22, r23 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r31, r32, r33 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    s2 = 2.0**0.5
    M = torch.zeros(*R.shape[:-2], 6, 6, dtype=torch.float64, device=R.device)

    # Upper-left 3x3 normal-normal block
    M[..., 0, 0], M[..., 0, 1], M[..., 0, 2] = r11 * r11, r12 * r12, r13 * r13
    M[..., 1, 0], M[..., 1, 1], M[..., 1, 2] = r21 * r21, r22 * r22, r23 * r23
    M[..., 2, 0], M[..., 2, 1], M[..., 2, 2] = r31 * r31, r32 * r32, r33 * r33

    # Upper-right 3x3 normal-shear block (scaled by sqrt(2))
    M[..., 0, 3], M[..., 0, 4], M[..., 0, 5] = s2 * r12 * r13, s2 * r11 * r13, s2 * r11 * r12
    M[..., 1, 3], M[..., 1, 4], M[..., 1, 5] = s2 * r22 * r23, s2 * r21 * r23, s2 * r21 * r22
    M[..., 2, 3], M[..., 2, 4], M[..., 2, 5] = s2 * r32 * r33, s2 * r31 * r33, s2 * r31 * r32

    # Lower-left 3x3 shear-normal block (scaled by sqrt(2))
    M[..., 3, 0], M[..., 3, 1], M[..., 3, 2] = s2 * r21 * r31, s2 * r22 * r32, s2 * r23 * r33
    M[..., 4, 0], M[..., 4, 1], M[..., 4, 2] = s2 * r11 * r31, s2 * r12 * r32, s2 * r13 * r33
    M[..., 5, 0], M[..., 5, 1], M[..., 5, 2] = s2 * r11 * r21, s2 * r12 * r22, s2 * r13 * r23

    # Lower-right 3x3 shear-shear block
    M[..., 3, 3] = r22 * r33 + r23 * r32
    M[..., 3, 4] = r21 * r33 + r23 * r31
    M[..., 3, 5] = r21 * r32 + r22 * r31

    M[..., 4, 3] = r12 * r33 + r13 * r32
    M[..., 4, 4] = r11 * r33 + r13 * r31
    M[..., 4, 5] = r11 * r32 + r12 * r31

    M[..., 5, 3] = r12 * r23 + r13 * r22
    M[..., 5, 4] = r11 * r23 + r13 * r21
    M[..., 5, 5] = r11 * r22 + r12 * r21

    return M.to(orig_dtype)


def build_so3_from_dip_azimuth(
    dip_deg: torch.Tensor,
    azimuth_deg: torch.Tensor,
) -> torch.Tensor:
    """
    Construct 3D rotation matrix R in SO(3) [..., 3, 3] from bedding dip theta and azimuth phi.
    """
    orig_dtype = dip_deg.dtype
    theta = torch.deg2rad(dip_deg.to(torch.float64))
    phi = torch.deg2rad(azimuth_deg.to(torch.float64))

    ct, st = torch.cos(theta), torch.sin(theta)
    cp, sp = torch.cos(phi), torch.sin(phi)
    zero = torch.zeros_like(ct)
    one = torch.ones_like(ct)

    # Azimuth rotation Rz(phi) followed by dip tilt Ry(theta)
    Rz = torch.stack(
        [
            torch.stack([cp, -sp, zero], dim=-1),
            torch.stack([sp, cp, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )
    Ry = torch.stack(
        [
            torch.stack([ct, zero, st], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-st, zero, ct], dim=-1),
        ],
        dim=-2,
    )
    R = torch.matmul(Rz, Ry)
    return R.to(orig_dtype)


def build_vti_kelvin_stiffness_tensor(
    vp0_kms: torch.Tensor,
    vs0_kms: torch.Tensor,
    rho_gcc: torch.Tensor,
    epsilon: torch.Tensor,
    delta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Construct the 6x6 symmetric VTI stiffness tensor C_VTI (in GPa) in the normalized
    Kelvin-Mandel basis from Thomsen parameters (Vp0, Vs0, rho, epsilon, delta, gamma).
    """
    orig_dtype = vp0_kms.dtype
    vp0 = vp0_kms.to(torch.float64)
    vs0 = vs0_kms.to(torch.float64)
    rho = rho_gcc.to(torch.float64)
    eps = epsilon.to(torch.float64)
    dlt = delta.to(torch.float64)
    gam = gamma.to(torch.float64)

    c33 = rho * (vp0**2)
    c55 = rho * (vs0**2)
    c44 = c55
    c11 = c33 * (1.0 + 2.0 * eps)
    c22 = c11
    c66 = c55 * (1.0 + 2.0 * gam)
    c12 = c11 - 2.0 * c66
    # Exact Thomsen C13 relation: (C13 + C55)^2 = (C33 - C55)^2 + 2 * delta * C33 * (C33 - C55)
    disc = torch.clamp((c33 - c55) ** 2 + 2.0 * dlt * c33 * (c33 - c55), min=1e-8)
    c13 = torch.sqrt(disc) - c55
    c23 = c13

    C = torch.zeros(*vp0.shape, 6, 6, dtype=torch.float64, device=vp0.device)
    C[..., 0, 0] = c11
    C[..., 1, 1] = c22
    C[..., 2, 2] = c33
    C[..., 0, 1] = c12
    C[..., 1, 0] = c12
    C[..., 0, 2] = c13
    C[..., 2, 0] = c13
    C[..., 1, 2] = c23
    C[..., 2, 1] = c23
    # In Kelvin-Mandel basis, shear diagonal entries are 2 * C44, 2 * C55, 2 * C66
    C[..., 3, 3] = 2.0 * c44
    C[..., 4, 4] = 2.0 * c55
    C[..., 5, 5] = 2.0 * c66
    return C.to(orig_dtype)


def compute_tti_phase_velocities(
    vp0_kms: torch.Tensor,
    vs0_kms: torch.Tensor,
    epsilon: torch.Tensor,
    delta: torch.Tensor,
    gamma: torch.Tensor,
    dip_deg: torch.Tensor,
    azimuth_deg: torch.Tensor,
    ray_direction_zyx: Tuple[float, float, float] = (0.5, 0.0, 0.8660254),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate exact Thomsen weak-anisotropy TTI phase velocities (Vp_TTI, Vsv_TTI, Vsh_TTI)
    for a propagation wavevector `ray_direction_zyx` relative to tilted bedding normal b(dip, azi):
        V_p(psi)  = V_p0 * (1 + delta * sin^2(psi) cos^2(psi) + epsilon * sin^4(psi))
        V_SV(psi) = V_s0 * (1 + (V_p0 / V_s0)^2 * (epsilon - delta) * sin^2(psi) cos^2(psi))
        V_SH(psi) = V_s0 * (1 + gamma * sin^2(psi))
    """
    theta = torch.deg2rad(dip_deg)
    phi = torch.deg2rad(azimuth_deg)

    # Symmetry axis unit normal b = (b_z, b_y, b_x)
    bz = torch.cos(theta)
    by = torch.sin(theta) * torch.sin(phi)
    bx = torch.sin(theta) * torch.cos(phi)

    nz, ny, nx = ray_direction_zyx
    n_norm = (nz * nz + ny * ny + nx * nx) ** 0.5 + 1e-12
    nz, ny, nx = nz / n_norm, ny / n_norm, nx / n_norm

    cos_psi = torch.clamp(torch.abs(bz * nz + by * ny + bx * nx), 0.0, 1.0)
    cos2 = cos_psi * cos_psi
    sin2 = torch.clamp(1.0 - cos2, 0.0, 1.0)
    sin4 = sin2 * sin2

    vp_tti = vp0_kms * (1.0 + delta * sin2 * cos2 + epsilon * sin4)
    vp_vs_sq = (vp0_kms / torch.clamp(vs0_kms, min=0.5)) ** 2
    vsv_tti = vs0_kms * (1.0 + vp_vs_sq * (epsilon - delta) * sin2 * cos2)
    vsh_tti = vs0_kms * (1.0 + gamma * sin2)
    return vp_tti, vsv_tti, vsh_tti


class ThomsenTTIElasticModulusEngine(nn.Module):
    """
    Coupled Multi-Parameter Anisotropic Elastic Modeler (Vp0, Vs0, rho, epsilon, delta, gamma)
    with Lie-algebraic so(6) Kelvin-Mandel stiffness tensor rotation across folded/faulted strata.
    """

    def __init__(self, num_lithologies: int = 8) -> None:
        super().__init__()
        self.num_lithologies = num_lithologies
        c = min(num_lithologies, len(ELASTIC_TTI_CATALOG))

        vp0 = [ELASTIC_TTI_CATALOG[i][1] for i in range(c)]
        vs0 = [ELASTIC_TTI_CATALOG[i][2] for i in range(c)]
        rho = [ELASTIC_TTI_CATALOG[i][3] for i in range(c)]
        eps = [ELASTIC_TTI_CATALOG[i][4] for i in range(c)]
        dlt = [ELASTIC_TTI_CATALOG[i][5] for i in range(c)]
        gam = [ELASTIC_TTI_CATALOG[i][6] for i in range(c)]

        self.register_buffer("litho_vp0", torch.tensor(vp0, dtype=torch.float32))
        self.register_buffer("litho_vs0", torch.tensor(vs0, dtype=torch.float32))
        self.register_buffer("litho_rho", torch.tensor(rho, dtype=torch.float32))
        self.register_buffer("litho_epsilon", torch.tensor(eps, dtype=torch.float32))
        self.register_buffer("litho_delta", torch.tensor(dlt, dtype=torch.float32))
        self.register_buffer("litho_gamma", torch.tensor(gam, dtype=torch.float32))

    @staticmethod
    def estimate_structural_dip_azimuth(vp_cube: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate local structural bedding dip (degrees in [0, 75]) and strike azimuth
        from the 3D spatial gradient of the stratigraphic velocity field.
        """
        # vp_cube: [..., Z, Y, X]
        v = vp_cube if vp_cube.ndim == 3 else vp_cube[0]
        gz = torch.zeros_like(v)
        gy = torch.zeros_like(v)
        gx = torch.zeros_like(v)

        gz[1:-1] = 0.5 * (v[2:] - v[:-2])
        gz[0] = v[1] - v[0]
        gz[-1] = v[-1] - v[-2]

        gy[:, 1:-1] = 0.5 * (v[:, 2:] - v[:, :-2])
        gy[:, 0] = v[:, 1] - v[:, 0]
        gy[:, -1] = v[:, -1] - v[:, -2]

        gx[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])
        gx[:, :, 0] = v[:, :, 1] - v[:, :, 0]
        gx[:, :, -1] = v[:, :, -1] - v[:, :, -2]

        horiz = torch.sqrt(gx * gx + gy * gy + 1e-12)
        vert = torch.abs(gz) + 1e-6
        dip_rad = torch.atan2(horiz, vert)
        dip_deg = torch.clamp(torch.rad2deg(dip_rad), 0.0, 75.0)
        azi_deg = torch.rad2deg(torch.atan2(gy, gx)) % 360.0
        if vp_cube.ndim == 4:
            return dip_deg.unsqueeze(0), azi_deg.unsqueeze(0)
        return dip_deg, azi_deg

    def forward(
        self,
        probs_post: torch.Tensor,
        vp_isotropic_kms: torch.Tensor,
        cavity_core: torch.Tensor | None = None,
        dip_deg: torch.Tensor | None = None,
        azimuth_deg: torch.Tensor | None = None,
        ray_direction_zyx: Tuple[float, float, float] = (0.5, 0.0, 0.8660254),
        chunk_z: int | None = 16,
    ) -> ElasticTTICubeBundle:
        """
        Synthesize coupled 3D elastic cubes (Vp0, Vs0, rho, Zp, Vp/Vs, Poisson) and
        Thomsen TTI anisotropic cubes (epsilon, delta, gamma, Vp_TTI, Vsv_TTI, Vsh_TTI)
        from post-triad lithology probabilities `probs_post` [B, C, Z, Y, X] and `vp_isotropic_kms`,
        operating along Z-slabs (`chunk_z=16`) to prevent multi-channel float32 OOMs on 128^3+ grids.
        """
        c = min(probs_post.shape[1], self.litho_vp0.numel())
        B, _, Z, Y, X = probs_post.shape
        vp0_cube = vp_isotropic_kms

        if dip_deg is None or azimuth_deg is None:
            dip_deg, azimuth_deg = self.estimate_structural_dip_azimuth(vp0_cube)

        step_z = Z if (chunk_z is None or int(chunk_z) <= 0) else max(1, min(Z, int(chunk_z)))

        vs0_cube = torch.empty_like(vp0_cube)
        rho_cube = torch.empty_like(vp0_cube)
        eps_cube = torch.empty_like(vp0_cube)
        dlt_cube = torch.empty_like(vp0_cube)
        gam_cube = torch.empty_like(vp0_cube)
        vp_tti = torch.empty_like(vp0_cube)
        vsv_tti = torch.empty_like(vp0_cube)
        vsh_tti = torch.empty_like(vp0_cube)
        vp_vs_ratio = torch.empty_like(vp0_cube)
        poisson_ratio = torch.empty_like(vp0_cube)
        zp_cube = torch.empty_like(vp0_cube)

        for z0 in range(0, Z, step_z):
            z1 = min(Z, z0 + step_z)
            p_slab = probs_post[:, :c, z0:z1]
            vp_slab = vp0_cube[:, z0:z1]

            vp0_prior = torch.einsum("bczyx,c->bzyx", p_slab, self.litho_vp0[:c])
            vs0_prior = torch.einsum("bczyx,c->bzyx", p_slab, self.litho_vs0[:c])
            rho_s = torch.einsum("bczyx,c->bzyx", p_slab, self.litho_rho[:c])
            eps_s = torch.einsum("bczyx,c->bzyx", p_slab, self.litho_epsilon[:c])
            dlt_s = torch.einsum("bczyx,c->bzyx", p_slab, self.litho_delta[:c])
            gam_s = torch.einsum("bczyx,c->bzyx", p_slab, self.litho_gamma[:c])

            scale_vp = vp_slab / torch.clamp(vp0_prior, min=1.2)
            vs0_s = torch.clamp(vs0_prior * (0.85 * scale_vp + 0.15), min=0.80, max=4.20)
            rho_gardner = 1.741 * (torch.clamp(vp_slab, min=1.5) ** 0.25)
            rho_s = torch.clamp(0.75 * rho_s + 0.25 * rho_gardner, min=1.85, max=3.10)

            if cavity_core is not None:
                cav_s = (cavity_core[:, 0, z0:z1] > 0.35).to(vp_slab.dtype)
                eps_s = eps_s * (1.0 - cav_s)
                dlt_s = dlt_s * (1.0 - cav_s)
                gam_s = gam_s * (1.0 - cav_s)

            vp_tti_s, vsv_tti_s, vsh_tti_s = compute_tti_phase_velocities(
                vp0_kms=vp_slab,
                vs0_kms=vs0_s,
                epsilon=eps_s,
                delta=dlt_s,
                gamma=gam_s,
                dip_deg=dip_deg[:, z0:z1],
                azimuth_deg=azimuth_deg[:, z0:z1],
                ray_direction_zyx=ray_direction_zyx,
            )

            vp_vs_s = vp_slab / torch.clamp(vs0_s, min=0.8)
            r2_s = vp_vs_s * vp_vs_s
            pois_s = torch.clamp(
                (r2_s - 2.0) / (2.0 * torch.clamp(r2_s - 1.0, min=0.25)), min=0.05, max=0.45
            )
            zp_s = rho_s * vp_tti_s

            vs0_cube[:, z0:z1] = vs0_s
            rho_cube[:, z0:z1] = rho_s
            eps_cube[:, z0:z1] = eps_s
            dlt_cube[:, z0:z1] = dlt_s
            gam_cube[:, z0:z1] = gam_s
            vp_tti[:, z0:z1] = vp_tti_s
            vsv_tti[:, z0:z1] = vsv_tti_s
            vsh_tti[:, z0:z1] = vsh_tti_s
            vp_vs_ratio[:, z0:z1] = vp_vs_s
            poisson_ratio[:, z0:z1] = pois_s
            zp_cube[:, z0:z1] = zp_s

        # Audit Theorem 2.7 (so(6) Kelvin-Mandel Stiffness Rotor Isometry) on a representative voxel slice
        with torch.no_grad():
            z_mid = vp0_cube.shape[1] // 2
            C_vti = build_vti_kelvin_stiffness_tensor(
                vp0_cube[:, z_mid],
                vs0_cube[:, z_mid],
                rho_cube[:, z_mid],
                eps_cube[:, z_mid],
                dlt_cube[:, z_mid],
                gam_cube[:, z_mid],
            ).to(torch.float64)
            R3 = build_so3_from_dip_azimuth(
                dip_deg[:, z_mid].to(torch.float64), azimuth_deg[:, z_mid].to(torch.float64)
            )
            M6 = build_kelvin_mandel_so6_rotor(R3)
            eye6 = torch.eye(6, device=M6.device, dtype=torch.float64)
            so6_ortho_err = float(torch.max(torch.abs(torch.matmul(M6.transpose(-1, -2), M6) - eye6)).item())
            C_tti = torch.matmul(torch.matmul(M6, C_vti), M6.transpose(-1, -2))

            tr_vti = torch.diagonal(C_vti, dim1=-2, dim2=-1).sum(dim=-1)
            tr_tti = torch.diagonal(C_tti, dim1=-2, dim2=-1).sum(dim=-1)
            trace_drift = float(torch.max(torch.abs(tr_tti - tr_vti)).item())

            frob_vti = torch.linalg.matrix_norm(C_vti, ord="fro")
            frob_tti = torch.linalg.matrix_norm(C_tti, ord="fro")
            frob_drift = float(torch.max(torch.abs(frob_tti - frob_vti)).item())

        return ElasticTTICubeBundle(
            vp0_kms=vp0_cube,
            vs0_kms=vs0_cube,
            rho_gcc=rho_cube,
            epsilon=eps_cube,
            delta=dlt_cube,
            gamma=gam_cube,
            dip_deg=dip_deg,
            azimuth_deg=azimuth_deg,
            vp_tti_kms=vp_tti,
            vsv_tti_kms=vsv_tti,
            vsh_tti_kms=vsh_tti,
            vp_vs_ratio=vp_vs_ratio,
            poisson_ratio=poisson_ratio,
            acoustic_impedance_zp=zp_cube,
            stiffness_trace_drift=trace_drift,
            stiffness_frobenius_drift=frob_drift,
            so6_orthogonality_error=so6_ortho_err,
        )
