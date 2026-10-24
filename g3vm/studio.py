"""
g3vm.studio: Interactive 3D WebGL/Canvas Geological Workstation, Portable HTML5 Studio Bundler,
and Publication Showcase Renderer (`v0.7.0`).

Provides Scenario 06 (`lunar_lava_tube_isru`) preset routing, Mode B Blind LOWO & Tuned Matérn-3/2 Kriging
3D property cubes (`blind_g3vm_vp`, `kriging_vp`, `fkriging_vp`, `blind_gain`), 6-quadrant 3D chair-cut &
vertical exaggeration controls, de-collided wellhead leader badges, and 2D fault/cavity contour overlays.
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
