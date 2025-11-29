# ROLT-phi-kappa3: Recursive Layer Optimality

Entropy-bounded recursive systems often peak at an intermediate depth. The ROLT-phi-kappa3 hypothesis models this with two depth profiles whose product forms the performance landscape:

* Coherence / gain `S_phi(d)`: typically a log-concave bump near the golden-ratio square (`phi^2 ~ 2.618`).
* Stability / capacity `S_kappa(d)`: a cubic-style saturation or decay with a mode in the 3-4 range.

We study `ell(d) = S_phi(d) * S_kappa(d)` and the optimal depth `d*` that maximizes it. This repository documents simulations, not a universal law.

---

## What this repo contains
- Figure generators for coherence, stability, and their product (`scripts/make_figures.py`).
- A depth-scaling illustration comparing classical vs quantum regimes (`scripts/depth_gap_plot.py`).
- Two QAOA MaxCut protocols for small 3-regular graphs:
  - Qiskit + Aer noise model (`scripts/qiskit_qaoa_protocol.py`).
  - PennyLane default.qubit vs default.mixed with depolarizing noise (`scripts/pennylane_qaoa_protocol.py`).
- Paper scaffold and a concise methods note (`paper/`).

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
python scripts/make_figures.py
```
Outputs: `figs/rolt_phi_kappa3_fig1.png`, `data/rolt_phi_kappa3_template.csv`.

Visualize the classical-vs-quantum depth gap:
```bash
python scripts/depth_gap_plot.py
```
Output: `figs/depth_gap.png`.

Run QAOA simulations (small graphs only; adjust `p_max`, `shots`, and `noise` as needed):
```bash
python scripts/qiskit_qaoa_protocol.py
python scripts/pennylane_qaoa_protocol.py
```

---

## Repository layout
```
README.md                Project overview and usage
LICENSE                  MIT
requirements.txt         Pip dependencies
env.yml                  Conda environment (python 3.11)
scripts/                 Plotting and simulation scripts
paper/                   LaTeX stub and methods note
data/, figs/             Created on demand by scripts
.gitignore
```

---

## Caveats
- Simulation-based: parameters are illustrative, not calibrated to hardware.
- Small-scale: graph sizes are tiny so brute-force MaxCut is feasible.
- Interpretive: ROLT-phi-kappa3 is a framing, not a theorem.

---

## License
MIT License (c) 2025
