"""
g3vm: DarkSnail-Geo3D Beltrami-Triad 3D Geological Velocity Modeler.

Exports the core mathematical modules, benchmark scenarios, and interactive viewer:
- `DarkSnailGeo3DVelocityModeler`
- `BeltramiWellLogShell`, `WellLogBeltramiResult`
- `ContinuousSpectralGeoField3D`, `RoPE3DMedoidPooler`
- `SkewSymmetricFaultTriad`, `SlowFastGeoCascade`
- `TemporalSurveyContinuitySplitter`, `ZenoSurgicalAnomalyGate`
- `NaiveSmoothVelocityBaseline`
- `compute_marginal_wasserstein1_vp`
- `build_geological_scenario`, `evaluate_scenario_comparison`
- `create_studio_app`, `export_portable_html_studio`, `render_leapfrog_showcase_figure`, `render_well_log_beltrami_figure`
"""

__version__ = "0.1.0"

from .engine import (
    BeltramiWellLogShell,
    ContinuousSpectralGeoField3D,
    DarkSnailGeo3DVelocityModeler,
    NaiveSmoothVelocityBaseline,
    RoPE3DMedoidPooler,
    SkewSymmetricFaultTriad,
    SlowFastGeoCascade,
    TemporalSurveyContinuitySplitter,
    WellLogBeltramiResult,
    ZenoSurgicalAnomalyGate,
    compute_marginal_wasserstein1_vp,
)
from .scenarios import (
    FAULT_MODE_NAMES,
    INDUSTRY_DEPLOYMENT_MATRIX,
    LITHOLOGY_CATALOG,
    SCENARIO_ALIASES,
    SCENARIO_Descriptions,
    build_geological_scenario,
    evaluate_scenario_comparison,
    resolve_scenario_name,
)
from .viewer import (
    create_studio_app,
    export_portable_html_studio,
    render_leapfrog_showcase_figure,
    render_well_log_beltrami_figure,
    resolve_static_dir,
    serialize_evaluation_payload,
)

__all__ = [
    "__version__",
    "DarkSnailGeo3DVelocityModeler",
    "BeltramiWellLogShell",
    "WellLogBeltramiResult",
    "ContinuousSpectralGeoField3D",
    "RoPE3DMedoidPooler",
    "SkewSymmetricFaultTriad",
    "SlowFastGeoCascade",
    "TemporalSurveyContinuitySplitter",
    "ZenoSurgicalAnomalyGate",
    "NaiveSmoothVelocityBaseline",
    "compute_marginal_wasserstein1_vp",
    "LITHOLOGY_CATALOG",
    "FAULT_MODE_NAMES",
    "INDUSTRY_DEPLOYMENT_MATRIX",
    "SCENARIO_ALIASES",
    "SCENARIO_Descriptions",
    "resolve_scenario_name",
    "build_geological_scenario",
    "evaluate_scenario_comparison",
    "resolve_static_dir",
    "create_studio_app",
    "export_portable_html_studio",
    "serialize_evaluation_payload",
    "render_leapfrog_showcase_figure",
    "render_well_log_beltrami_figure",
]
