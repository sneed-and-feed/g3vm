"""
g3vm.studio: Interactive 3D WebGL/Canvas Geological Workstation, Portable HTML5 Studio Bundler,
and Publication Showcase Renderer (`v0.4.0`).

Provides Scenario 06 (`lunar_lava_tube_isru`) preset routing, `g3vm_eps_r` (`Dielectric Permittivity εr (Radar)`)
and `g3vm_v_em` (`Radar Velocity v_EM (m/ns)`) 3D field rendering, and borehole Beltrami spike callouts
for `LSPE-GT1`, `VIPER-TR2`, and `LRS-IL3`.
"""

from __future__ import annotations

from .viewer import (
    create_studio_app,
    export_portable_html_studio,
    render_leapfrog_showcase_figure,
    render_well_log_beltrami_figure,
    resolve_static_dir,
    serialize_evaluation_payload,
)

__all__ = [
    "create_studio_app",
    "export_portable_html_studio",
    "render_leapfrog_showcase_figure",
    "render_well_log_beltrami_figure",
    "resolve_static_dir",
    "serialize_evaluation_payload",
]
