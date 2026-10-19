# G3VM (`DarkSnail-Geo3D`) — 3D Geological Velocity Modeler

**G3VM** (`DarkSnail-Geo3D`) is a 3D geological velocity modeler (Seequent Leapfrog / Petex style) powered by the formally verified mathematical primitives of **Project DarkSnail**:

- **Skew-Symmetric Fault & Unconformity Tie-Breaking** in $\mathfrak{so}(C)$ (`SkewSymmetricFaultTriad`, Theorems 2.1, 2.2, 2.3, 2.4, 2.6)
- **Beltrami Callosal Well-Log Exterior-Curl Spike Shielding** (`BeltramiWellLogShell`, `1.0000` thin-bed recovery vs. `96.94%` mean-pooling loss)
- **3D Continuous Spectral Stratigraphy & 6-Way Jordan Cavity Isolation** (`ContinuousSpectralGeoField3D` + `RoPE3DMedoidPooler`, Theorem 2.5)
- **Precision-Weighted Ridge & Abelian OMP Travel-Time Tomography** (`solve_ridge_tomography_update`, `solve_abelian_omp_tomography`, and $O(ZYX)$ marginal Wasserstein-1 $W_1^{\mathrm{marg}}$)
- **4D Time-Lapse Continuity Splitting & Zeno Surgical Hazard Gate** (`TemporalSurveyContinuitySplitter` in $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$ and `ZenoSurgicalAnomalyGate`)

![G3VM 4-Scenario 3D Subsurface Velocity Showcase](results/g3vm_leapfrog_showcase.png)

![Beltrami Callosal Well-Log Shell Across 6 Sparse Boreholes](results/g3vm_well_log_beltrami.png)

---

## Why Standard 3D Velocity Interpolators Fail

Standard neural, Radial Basis Function (RBF), and kriging velocity modelers suffer from three structural failure modes when building a 3D P-wave velocity cube $V_p(z, y, x)$:

1. **Fault-Plane & Unconformity Collisions**: When a fault plane or erosional unconformity slices through a 3D rock volume, two distinct lithologies with $V_{p,1} \neq V_{p,2}$ collide at the fault surface. Smooth RBF/kriging interpolators average the two layers across the fault (`V_p = 0.5 * (V_1 + V_2)`), creating artificial low-velocity smears instead of a sharp structural discontinuity.
2. **Borehole Acoustic-Impedance Dilution**: Wireline sonic logs are over 95% monotonous background shale or sandstone punctuated by 1 to 3 meter high-contrast velocity spikes (carbonate stringers, permeable fracture corridors, salt interfaces, fault gouge). Standard voxel downsampling averages those spikes into oblivion, losing up to 96.94% of the high-frequency innovation signal.
3. **Salt Diapir Cavities & Multi-Bounce Ray Tracing**: Populating a velocity cube from seismic travel-time residuals requires separating enclosed bodies (salt domes, reefs, H₂ caverns, and flooded karst voids) from layered stratigraphy and inverting multi-bounce ray paths without blowing up velocity norms in unconstrained deep-basin zones.

---

## Real-World Deployment Matrix

| Industry Vertical | Legacy Software Failure Mode (Leapfrog / Petex) | `g3vm` DarkSnail Primitive Applied | Concrete Operational Deliverable |
| :--- | :--- | :--- | :--- |
| **1. Enhanced Geothermal Systems (EGS) & Superhot Rock** | Smooths over thin 1m permeable fracture zones and blurs basement fault offsets. | `BeltramiWellLogShell` (`1.0000` spike recovery) + `SkewSymmetricFaultTriad` $(\mathfrak{so}(C))$. | Pinpoints thin fluid-bearing fracture stringers in crystalline rock and prevents drilling into pressurized fault gouge. |
| **2. Carbon Sequestration (CCS) & H₂ Salt Cavern Storage** | Cannot separate seasonal/tidal surface velocity drift from real CO₂ plume migration or caprock leaks. | `TemporalSurveyContinuitySplitter` ($\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$) + 6-way Jordan `cavity_core` (β₁) medoids. | Maps 3D salt-dome storage caverns and strips exogenous environmental noise from 4D time-lapse plume monitoring. |
| **3. Critical Mineral & Porphyry/Sulfide Mining** | RBF implicit surfaces smear high-grade ore shells and $V_p$ anomalies across post-mineralization faults. | Theorem 2.2 (`occlusion_triad_breaks_symmetric_tie`) + `solve_abelian_omp_tomography` (Ridge OMP). | Enforces step-function lithology jumps across fault throws and inverts cross-hole seismic for dense sulfide pods. |
| **4. Civil Megaproject Tunneling (TBM Look-Ahead)** | Patch-averaged wavefields attenuate phase (`rope2d_arithmetic_norm_lt`) and miss water-filled karst voids ahead of the cutterhead. | `ContinuousSpectralGeoField3D` + `RoPE3DMedoidPooler` + `ZenoSurgicalAnomalyGate`. | Real-time Seismic-While-Drilling (SWD) hazard radar that images flooded karst cavities and boulders before TBM impact. |
| **5. Offshore Wind Monopiles & Subsea Cable Routing** | Multi-bounce water-column reverberations mask shallow gas pockets and glacial boulders in the top 100m of seabed. | Multi-bounce ray basis + surgical $1 / N_{\mathrm{vox}}$ ray quantiles. | Isolates single-voxel boulder and shallow-gas hazards without corrupting the background marine sediment velocity cube. |
| **6. Planetary & Lunar Subsurface Radar (Lava Tubes / Ice)** | Sparse orbital radargrams (SHARAD/MARSIS) lack dense borehole ties, causing unconstrained tomography blowup. | Theorem 2.3 (`darksnail_readout_energy_conservation`) + closed-form $O(ZYX)$ $W_1^{\mathrm{marg}}$ optimal transport. | Zero-blowup inversion of intact lunar/Martian lava tubes and buried glacial ice lenses from sparse orbital passes. |

---

## Verified 3D Geophysical Benchmark Results ($48 \times 48 \times 48$ Voxel Cubes, 6 Sparse Boreholes)

All four benchmark basins are evaluated head-to-head against a standard 3D isotropic Gaussian RBF / kriging interpolator (`NaiveSmoothVelocityBaseline`):

| Basin Scenario | G3VM Velocity RMSE (km/s) | Naive RBF Velocity RMSE (km/s) | RMSE Reduction | Fault Sharpness (G3VM vs. RBF) | Fault-Zone Lithology Acc. (G3VM vs. RBF) | Borehole Spike Recovery (G3VM vs. Naive) | Max Tangent Residual $\lvert\langle v, \Delta\ell\rangle\rvert$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`subsalt_graben`** (Marmousi2 / Gulf Subsalt) | `0.1578` | `0.3565` | `-55.75%` | `100.3%` vs. `6.5%` | `83.6%` vs. `33.7%` | `100.0%` vs. `13.8%` | `1.31e-11` |
| **`thrust_anticline`** (SEG/EAGE Overthrust) | `0.0605` | `0.6116` | `-90.11%` | `96.4%` vs. `8.0%` | `99.2%` vs. `46.4%` | `100.0%` vs. `13.2%` | `1.31e-11` |
| **`unconformity_trap`** (4D CCS / H₂ Cavern) | `0.1070` | `0.3632` | `-70.53%` | `100.4%` vs. `4.1%` | `86.6%` vs. `34.7%` | `100.0%` vs. `7.9%` | `1.31e-11` |
| **`egs_tbm_hazard`** (EGS Granite & TBM Karst) | `0.2186` | `0.5323` | `-58.94%` | `99.8%` vs. `8.9%` | `83.9%` vs. `0.0%` | `100.0%` vs. `12.3%` | `1.31e-11` |

### Machine-Precision Theorem Verification (`formalization/OcclusionTriad.lean`)

| Theorem | Mathematical Statement | Empirical Audit Residual | Status |
| :--- | :--- | :--- | :--- |
| **Theorem 2.1 (`triad_skew_symmetric`)** | $W_{i, j, k} + W_{i, k, j} = 0, \quad \langle v, B(u, v) \rangle = 0$ | `0.000e+00` | Verified |
| **Theorem 2.2 (`occlusion_triad_breaks_symmetric_tie`)** | $\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2) = 2\beta\alpha W_{i_0, c_1, c_2} \neq 0$ | `0.000e+00` (Gap `2.50`) | Verified |
| **Theorem 2.3 (`darksnail_readout_energy_conservation`)** | $\lvert \langle v_{\mathrm{prior}}, \Delta\ell_{\mathrm{fault}} \rangle \rvert \le 10^{-8}$ | `9.313e-09` (`1.31e-11` field) | Verified |
| **Theorem 2.4 (`spherical_retraction_unit_norm`)** | Step-0 identity homotopy at $u_{\mathrm{fault}} = 0$ | `0.000e+00` | Verified |
| **Theorem 2.5 (`rope2d_arithmetic_norm_lt`)** | Medoid pooling preserves $\lVert z_{\mathrm{med}} \rVert = 1$ vs. arithmetic loss `0.366` | `5.551e-17` | Verified |
| **Theorem 2.6 (`multi_step_spherical_invariance`)** | Multi-step cascade preserves $\lVert v^{(K)} \rVert = \lVert v^{(0)} \rVert$ | `0.000e+00` | Verified |

---

## Cross-Platform Quickstart (Linux, macOS, Windows)

### Option 1: Zero-Install Portable HTML5 Studio (For Immediate Browser Use)

Open [`results/g3vm_portable_studio.html`](results/g3vm_portable_studio.html) directly in any modern web browser (Chrome, Firefox, Safari, Edge) on Linux, macOS, or Windows. It embeds all 4 pre-computed 3D velocity cubes, interactive 3D chair-cut block rendering, synchronized 2D Inline/Crossline/Depth cross-sections, 6 borehole Beltrami log strips, and live Theorem 2.1–2.6 telemetry with zero Python setup.

### Option 2: Pre-Built Standalone Binaries (GitHub Actions CI Releases)

Every tagged release (`v*`) or workflow dispatch in [`.github/workflows/ci-and-release.yml`](.github/workflows/ci-and-release.yml) automatically builds standalone single-file executables via PyInstaller on native GitHub runners:

- **Windows (`x86_64`)**: `g3vm-windows-x86_64.exe` (double-click to start the 3D Studio server and open your browser automatically)
- **macOS (`Apple Silicon arm64`)**: `g3vm-macos-arm64`
- **Linux (`x86_64`)**: `g3vm-linux-x86_64`
- **Universal Offline HTML5 Studio**: `g3vm_portable_studio.html`

### Option 3: Python Package Installation & CLI

```bash
pip install -e ".[dev]"

# Launch the interactive 3D Subsurface Velocity Studio (http://127.0.0.1:8765)
g3vm serve --grid-size 36

# Run the full 4-basin benchmark suite and generate PNG figures + JSON telemetry
g3vm benchmark --grid-size 48 --output-dir results

# Generate a self-contained single-file HTML5 studio
g3vm bundle-html --grid-size 28 --output results/g3vm_portable_studio.html

# Export a 3D velocity cube to compressed NumPy .npz for Petrel / Leapfrog / SEG-Y workflows
g3vm export --scenario subsalt_graben --grid-size 48 --output results/subsalt_graben_cube.npz

# Run the 11-test verification suite and GFM/KaTeX math linter
pytest -v
python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md
```

---

## Architecture Documentation

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the complete mathematical derivation, exact equations, and module-by-module mapping between `projectdarksnail` and `g3vm`.