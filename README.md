# ROLT-phi-kappa3: Recursive Layer Optimality

Entropy-bounded recursive systems often peak at an intermediate depth. The ROLT-phi-kappa3 hypothesis models this with two depth profiles whose product forms the performance landscape:

* Coherence / gain `S_phi(d)`: typically a log-concave bump near the golden-ratio square (`phi^2 ~ 2.618`).
* Stability / capacity `S_kappa(d)`: a cubic-style saturation or decay with a mode in the 3-4 range.

We study `ell(d) = S_phi(d) * S_kappa(d)` and the optimal depth `d*` that maximizes it. This repository documents simulations, not a universal law.

---

## What this repo contains

**Figure scripts** (`scripts/figures/`):
- `plot_phi_kappa3_profile.py` — coherence, stability, and their product overlay; writes `figs/rolt_phi_kappa3_fig1.png` and `data/rolt_phi_kappa3_template.csv`.
- `plot_depth_scaling_gap.py` — classical vs quantum depth-scaling gap illustration; writes `figs/depth_gap.png`.

**QAOA simulations** (`scripts/qaoa/`):
- `qaoa_maxcut_qiskit_aer.py` — QAOA MaxCut sweep on 3-regular graphs using Qiskit + Aer depolarizing noise model.
- `qaoa_maxcut_pennylane.py` — same sweep using PennyLane `default.qubit` (ideal) vs `default.mixed` (depolarizing noise).

**Dirac / APS lattice simulations** (`scripts/dirac/`):
- `rolt_dirac_hopping_options.py` — compares four hopping-term variants (standard, non-Hermitian i!, real |i!|, single-layer baseline) on the ROLT potential.
- `rolt_aps_eta_3case.py` — APS-style eta and spectral gap for three Dirac model cases (non-Hermitian, domain-wall, hybrid+twist).
- `rolt_aps_eta_5case_sweep.py` — corrective rewrite: five cases, fixed phases, proper Hermiticity checks, low-energy eta window, and m0 / lambda parameter sweeps; writes `figs/rolt_aps_v2_full_spectra.png`.

**Paper scaffold** (`paper/`): LaTeX stub and concise methods note.

**Simulation outputs** (`outputs/`): figures and data files written by scripts (not committed; regenerated on demand).

---

## Conceptual frame
1. **Competing effects**: deeper recursion can increase coherence/expressivity while also accumulating noise/entropy/cost.
2. **Product profile**: if both `S_phi` and `S_kappa` are unimodal or log-concave near their modes, their product is unimodal with a single maximizer.
3. **Non-Gaussian variants**: skewed bells, stretched exponentials, `sech^2`, and light heavy-tails also produce a well-behaved product under mild regularity.

---

## Quick start
Install dependencies (pip or conda):
```bash
pip install -r requirements.txt
# or
conda env create -f env.yml && conda activate rolt
```

Generate the coherence/stability/product figure and template CSV:
```bash
python scripts/figures/plot_phi_kappa3_profile.py
```
Outputs: `outputs/figures/rolt_phi_kappa3_fig1.png`, `outputs/data/rolt_phi_kappa3_template.csv`.

Visualize the classical-vs-quantum depth gap:
```bash
python scripts/figures/plot_depth_scaling_gap.py
```
Output: `outputs/figures/depth_gap.png`.

Run QAOA simulations (small graphs only; adjust `p_max`, `shots`, and `noise` as needed):
```bash
python scripts/qaoa/qaoa_maxcut_qiskit_aer.py
python scripts/qaoa/qaoa_maxcut_pennylane.py
```

Run Dirac / APS lattice simulations:
```bash
python scripts/dirac/rolt_dirac_hopping_options.py
python scripts/dirac/rolt_aps_eta_3case.py
python scripts/dirac/rolt_aps_eta_5case_sweep.py
```

---

## Repository layout
```
README.md                Project overview and usage
LICENSE                  MIT
requirements.txt         Pip dependencies
env.yml                  Conda environment (python 3.11)
scripts/
  figures/               Plotting scripts (profile, depth gap)
  qaoa/                  QAOA MaxCut simulations (Qiskit, PennyLane)
  dirac/                 Dirac operator / APS eta simulations
paper/                   LaTeX stub and methods note
outputs/
  figures/               Generated plots (PNG)
  data/                  Generated data (CSV)
results.md               Summary of simulation results
nextsteps.md             Planned extensions and open tasks
.gitignore
```

---

## Caveats
- Simulation-based: parameters are illustrative, not calibrated to hardware.
- Small-scale: graph sizes are tiny so brute-force MaxCut is feasible.
- Interpretive: ROLT-phi-kappa3 is a framing, not a theorem.

---
##message: "If you use this software, please cite it as below."
```
title: "ROLT-phi-kappa3: Recursive Layer Optimality"
authors:
  - family-names: "Farkas"
    given-names: "Zolton"
version: "0.1.0"
license: "MIT"
```
## License
MIT License (c) 2025

## Complete HTML readers


Download a reader below and open the HTML file in a browser. Each file contains every original page, selectable extracted text where available, a cascading reveal, and **Export GIF** with the EncapsulatorP logo on every frame. All reading assets are included, so the downloaded reader works offline.

The source documents keep their original arguments, qualifications, attribution and reference status. The reading effect does not change those claims. The original print files remain recoverable from the pinned source revisions recorded in [reading-editions.json](reading-editions.json).

| Complete reader | Original pages |
| --- | ---: |
| [rolt](paper/rolt.html) | 4 |

To reproduce the reading editions, use the pinned source revisions and the [shared HTML importer](https://github.com/EncapsulatorP/EncapsulatorP.github.io/blob/main/atlas/import_pdf_readers.py). The manifest records the original byte hashes and page coverage.
