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
- `read_las2_file`, `write_las2_file`, `read_dev_survey`, `write_dev_survey`, `read_segy_cube`, `write_segy_cube`, `stream_segy_volume`, `beltrami_downsample_3d_cube`
- `NaiveSmoothVelocityBaseline`
- `compute_marginal_wasserstein1_vp`
- `build_geological_scenario`, `evaluate_scenario_comparison`
- `create_studio_app`, `export_portable_html_studio`, `render_leapfrog_showcase_figure`, `render_well_log_beltrami_figure`
"""

__version__ = "0.5.0"

from .anisotropy import (
    DielectricCubeBundle,
    ELASTIC_TTI_CATALOG,
    ElasticTTICubeBundle,
    LUNAR_DIELECTRIC_CATALOG,
    LUNAR_ELASTIC_TTI_CATALOG,
    OlhoeftStrangwayDielectricEngine,
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
    beltrami_downsample_3d_cube,
    ensure_lunar_lava_tube_field_files,
    ensure_utah_forge_field_files,
    rasterize_deviated_well_into_3d_volume,
    read_dev_survey,
    read_las2_file,
    read_segy_cube,
    sample_3d_cube_along_trajectory,
    solve_minimum_curvature_trajectory,
    stream_segy_volume,
    write_dev_survey,
    write_las2_file,
    write_segy_cube,
)
from .scenarios import (
    FAULT_MODE_NAMES,
    INDUSTRY_DEPLOYMENT_MATRIX,
    LITHOLOGY_CATALOG,
    LUNAR_LITHOLOGY_CATALOG,
    SCENARIO_ALIASES,
    SCENARIO_Descriptions,
    SCENARIO_LITHOLOGY_CATALOGS,
    build_geological_scenario,
    build_lunar_lava_tube_isru_scenario,
    evaluate_scenario_comparison,
    resolve_scenario_name,
)
from .blind_benchmark import (
    evaluate_beltrami_noise_and_washout_ablation,
    evaluate_blind_holdout_and_baselines,
    fit_and_predict_anisotropic_matern_kriging,
    run_referee_audit_suite,
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
    "LUNAR_ELASTIC_TTI_CATALOG",
    "LUNAR_DIELECTRIC_CATALOG",
    "ElasticTTICubeBundle",
    "DielectricCubeBundle",
    "ThomsenTTIElasticModulusEngine",
    "OlhoeftStrangwayDielectricEngine",
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
    "stream_segy_volume",
    "beltrami_downsample_3d_cube",
    "ensure_utah_forge_field_files",
    "ensure_lunar_lava_tube_field_files",
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
    "LUNAR_LITHOLOGY_CATALOG",
    "SCENARIO_LITHOLOGY_CATALOGS",
    "FAULT_MODE_NAMES",
    "INDUSTRY_DEPLOYMENT_MATRIX",
    "SCENARIO_ALIASES",
    "SCENARIO_Descriptions",
    "resolve_scenario_name",
    "build_geological_scenario",
    "build_lunar_lava_tube_isru_scenario",
    "evaluate_scenario_comparison",
    "fit_and_predict_anisotropic_matern_kriging",
    "evaluate_blind_holdout_and_baselines",
    "evaluate_beltrami_noise_and_washout_ablation",
    "run_referee_audit_suite",
    "resolve_static_dir",
    "create_studio_app",
    "export_portable_html_studio",
    "serialize_evaluation_payload",
    "render_leapfrog_showcase_figure",
    "render_well_log_beltrami_figure",
]
