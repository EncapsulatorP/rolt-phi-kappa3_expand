# ROLT-φκ³ Hypothesis: Recursive Layer Optimality & Symbolic Attractor Convergence

## Overview
The **ROLT‑φκ³ Hypothesis** proposes that entropy‑bounded recursive systems exhibit an **optimal depth** \(d^*\) where the trade‑off between coherence gain and stability cost is maximized. The “sweet spot” emerges from two interacting components:

- **Golden‑ratio coherence scaling** (\(\varphi^2 \approx 2.618\)): a natural spectral feature of many recursive structures.  
- **\(\kappa^3\) stability attractor** (mode \(\mu_\kappa\) typically in the 3–4 range): a cubic‑saturation effect reflecting noise, cost, or entropy constraints.

We model depth‑dependent performance as the multiplicative profile
\[
\ell(d) = S_\phi(d)\, S_\kappa(d),
\]
with a unique maximum \(d^*\) under mild unimodality/log‑concavity assumptions.

> **Important:** This is a **hypothesis**, not a theorem or universal law. The repo documents **fun simulations** and the conclusions we draw from them.

---

## Conceptual Framework

### 1) Recursive layer optimization
Most recursive systems face two competing effects:
- **Constructive gain** — deeper layers initially improve approximation/representation/coherence.
- **Dissipative cost** — beyond some point, error, noise, or entropy growth reduces performance.

We formalize with two depth profiles:
- $$\(S_\phi(d)\)$$: coherence/gain (often modeled as a Gaussian‑like or stretched‑exponential bump centered near \(\varphi^2\)).
- $$\(S_\kappa(d)\)$$: stability/capacity (often modeled as a $$sech\(^2\)$$, stretched‑exponential, or heavy‑tailed decay centered near $$\(\mu_\kappa\))$$.

Their product $$\(\ell(d)\)$$ identifies the **effective performance landscape** and the **optimal depth** $$\(d^*\)$$.

### 2) Why an optimal depth exists
An optimum naturally arises when **marginal gain equals marginal loss**:
\[
$$\frac{d}{dd}\,\ell(d) = 0.$$
\]
In simulations of layered quantum circuits, deep networks under constraints, and symbolic recursions, performance peaks at intermediate depths rather than growing indefinitely.

### 3) Non‑Gaussian generalization
We also study non‑Gaussian bands: skewed bells, Tsallis/Student‑t (heavy tails), and soliton‑like $$\(\operatorname{sech}^2\)$$. Under **strict log‑concavity** (or quasi log‑concavity) near their modes and regular tails, the product remains **unimodal** and yields a unique maximizer.

---

## Applications (simulation‑based)

1. **Quantum algorithms (QAOA‑style circuits)**  
   Simulations with realistic noise models often show optimal depths in the **2–5** range. Beyond that, accumulated noise outweighs coherence gains.

2. **Deep learning (resource‑bounded)**  
   When data/compute are limited, shallow‑to‑intermediate depth models can outperform much deeper ones. The hypothesis provides a compact way to reason about this balance.

3. **Symbolic AI / recursive reasoning**  
   Multi‑step planners face combinatorial blow‑up. The hypothesis suggests a principled cutoff depth for efficient reasoning.

4. **Biological & synthetic branching**  
   Resource‑limited branching processes naturally terminate after a few generations; simulations show this cutoff can be captured by $$\(\phi\)–\(\kappa^3\)$$ balancing.

> We report only **our simulation results** here. External hardware/platform names and third‑party datasets are intentionally excluded.

---

## Strengths
- **Cross‑domain framing:** unifies quantum/AI/biological recursion under a single lens.  
- **Interpretable:** clean separation of gain $$(\(\phi\))$$ and cost $$(\(\kappa^3\))$$.  
- **Falsifiable:** predicts a measurable $$\(d^*\)$$; simulations can support or refute.

## Limitations
- **Not universal:** some systems may lack $$\(\phi\)$$‑like scaling or cubic‑style saturation.  
- **Simulation‑dependent:** profiles are fitted; not (yet) first‑principles derivations.  
- **Scope:** intended for **entropy‑bounded** recursion, not exact infinite fractals or unconstrained deep nets.

---

## Roadmap

**Phase 1 — Simulation validation**  
Generate depth‑performance curves across quantum, AI, and symbolic recursion simulations. Fit $$\(S_\phi\)$$, $$\(S_\kappa\)$$, and locate $$\(d^*\)$$.

**Phase 2 — Non‑Gaussian bands**  
Test skewed/heavy‑tailed/soliton profiles; verify uniqueness conditions and robustness of $$\(d^*\)$$.

**Phase 3 — Comparative studies**  
Compare predicted vs. simulated optima across tasks and scales. Preregister simple tests where possible.

**Phase 4 — Abstraction layer**  
Explore symbolic attractors and entropy‑geometry embeddings as higher‑level explanations for observed cutoffs.

---

## Getting Started

### Install
```bash
pip install -r requirements.txt
# or
conda env create -f env.yml && conda activate rolt
```

### Generate figures (coherence, stability, product)
```bash
python scripts/make_figures.py
```
Outputs plots under `figs/` and a starter CSV in `data/`.

### Depth‑gap illustration (classical vs quantum scaling)
```bash
python scripts/depth_gap_plot.py
```

---

## Repo Structure
```
.
├── README.md
├── LICENSE
├── requirements.txt
├── env.yml
├── .gitignore
├── data/
│   └── rolt_phi_kappa3_template.csv
├── figs/
│   ├── rolt_phi_kappa3_fig1.png
│   └── depth_gap.png
├── scripts/
│   ├── make_figures.py
│   └── depth_gap_plot.py
└── paper/
    ├── rolt_phi_kappa3.tex
    └── methods_onepager.md
```

---

## License
MIT License © 2025

## Notes
This repository reflects a **hypothesis‑driven, simulation‑based exploration**. It is not positioned as a theorem or universal law.
