"""
g3vm: DarkSnail-Geo3D Beltrami-Triad 3D Geological Velocity Modeler.

Exports the core mathematical modules, TTI anisotropy & elastic engine, industry LAS/SEG-Y I/O,
benchmark scenarios, and interactive viewer:
- `DarkSnailGeo3DVelocityModeler`
- `BeltramiWellLogShell`, `WellLogBeltramiResult`
- `ContinuousSpectralGeoField3D`, `RoPE3DMedoidPooler`
- `SkewSymmetricFaultTriad`, `SlowFastGeoCascade`
- `TemporalSurveyContinuitySplitter`, `ZenoSurgicalAnomalyGate`
- `ThomsenTTIElasticModulusEngine`, `ElasticTTICubeBundle`, `ELASTIC_TTI_CATALOG`
- `DeviatedWellTrajectory`, `LAS2WellRecord`, `solve_minimum_curvature_trajectory`
- `read_las2_file`, `write_las2_file`, `read_dev_survey`, `write_dev_survey`, `read_segy_cube`, `write_segy_cube`
- `NaiveSmoothVelocityBaseline`
- `compute_marginal_wasserstein1_vp`
- `build_geological_scenario`, `evaluate_scenario_comparison`
- `create_studio_app`, `export_portable_html_studio`, `render_leapfrog_showcase_figure`, `render_well_log_beltrami_figure`
"""

__version__ = "0.2.0"

from .anisotropy import (
    ELASTIC_TTI_CATALOG,
    ElasticTTICubeBundle,
    ThomsenTTIElasticModulusEngine,
    build_kelvin_mandel_so6_rotor,
    build_so3_from_dip_azimuth,
    build_vti_kelvin_stiffness_tensor,
    compute_tti_phase_velocities,
)
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
from .io import (
    DeviatedWellTrajectory,
    LAS2WellRecord,
    rasterize_deviated_well_into_3d_volume,
    read_dev_survey,
    read_las2_file,
    read_segy_cube,
    sample_3d_cube_along_trajectory,
    solve_minimum_curvature_trajectory,
    write_dev_survey,
    write_las2_file,
    write_segy_cube,
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
    "ELASTIC_TTI_CATALOG",
    "ElasticTTICubeBundle",
    "ThomsenTTIElasticModulusEngine",
    "build_kelvin_mandel_so6_rotor",
    "build_so3_from_dip_azimuth",
    "build_vti_kelvin_stiffness_tensor",
    "compute_tti_phase_velocities",
    "DeviatedWellTrajectory",
    "LAS2WellRecord",
    "solve_minimum_curvature_trajectory",
    "sample_3d_cube_along_trajectory",
    "rasterize_deviated_well_into_3d_volume",
    "read_las2_file",
    "write_las2_file",
    "read_dev_survey",
    "write_dev_survey",
    "read_segy_cube",
    "write_segy_cube",
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
