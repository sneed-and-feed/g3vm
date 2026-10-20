"""
g3vm.geophysics: Coupled Seismic-Dielectric Petrophysical Engine (`OlhoeftStrangwayDielectricEngine`),
Thomsen TTI Anisotropic Elastic Modeler (`ThomsenTTIElasticModulusEngine`), Beltrami-Triad 3D Velocity
Modeler (`DarkSnailGeo3DVelocityModeler`), and Scenario 06 (`build_lunar_lava_tube_isru_scenario`).

Operationalizes Row 6 of the Real-World Deployment Matrix (`Planetary & Lunar Subsurface Radar /
Lava Tubes / Ice`) anchored to NASA PDS Apollo 17 LSPE (`urn:nasa:pds:a17lspe`), SELENE Kaguya LRS
(`5 MHz` two-peak lava tube echo, Kaku et al. 2017), LRO Mini-RF (Carrer et al. 2024), and
VIPER-TRIDENT cryogenic water-ice targets (Horvath et al. 2022).
"""

from __future__ import annotations

from .anisotropy import (
    ELASTIC_TTI_CATALOG,
    LUNAR_DIELECTRIC_CATALOG,
    LUNAR_ELASTIC_TTI_CATALOG,
    DielectricCubeBundle,
    ElasticTTICubeBundle,
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

__all__ = [
    "ELASTIC_TTI_CATALOG",
    "LUNAR_ELASTIC_TTI_CATALOG",
    "LUNAR_DIELECTRIC_CATALOG",
    "DielectricCubeBundle",
    "ElasticTTICubeBundle",
    "OlhoeftStrangwayDielectricEngine",
    "ThomsenTTIElasticModulusEngine",
    "build_kelvin_mandel_so6_rotor",
    "build_so3_from_dip_azimuth",
    "build_vti_kelvin_stiffness_tensor",
    "compute_tti_phase_velocities",
    "BeltramiWellLogShell",
    "WellLogBeltramiResult",
    "ContinuousSpectralGeoField3D",
    "RoPE3DMedoidPooler",
    "SkewSymmetricFaultTriad",
    "SlowFastGeoCascade",
    "TemporalSurveyContinuitySplitter",
    "ZenoSurgicalAnomalyGate",
    "DarkSnailGeo3DVelocityModeler",
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
    "build_lunar_lava_tube_isru_scenario",
    "build_geological_scenario",
    "evaluate_scenario_comparison",
]
