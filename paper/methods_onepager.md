# Methods: Simulation Protocols for the ROLT-φκ³ 

This note documents **our simulations** used to explore  ROLT‑φκ³ . We do not claim universality; we report what we observed in controlled experiments.

## 1. QAOA-style Quantum Circuits

### Task
MaxCut on connected 3-regular graphs with n nodes (n even ≥ 4).

### Depth
QAOA depth \(d=p\) (each layer = cost + mixer unitaries).

### Observables
- **Noiseless approximation ratio** \(\alpha(d)\): expected cut value under the ideal distribution divided by the max cut (computed by brute force for small n).
- **Noisy fidelity proxy** \(F(d)\): Hellinger affinity between ideal and noisy bitstring histograms.

### Models
- **Coherence/Gain** \(S_\phi(d)\): Gaussian-like or stretched-exponential bump centered near \(\varphi^2\).
- **Stability/Capacity** \(S_\kappa(d)\): \(\operatorname{sech}^2\) or heavy-tailed decay centered near \(\mu_\kappa\) in the 3–4 range.

### Protocol
1. Generate a random connected 3-regular graph (n=8 by default).
2. Sweep depths \(d=1\ldots 10\).
3. Collect \(\alpha(d)\) from an ideal simulator and \(F(d)\) from a depolarizing-noise simulator.
4. Fit \(S_\phi(d)\) and \(S_\kappa(d)\) (optionally) and analyze \(\ell(d)=S_\phi S_\kappa\).
5. Identify any optimal depth \(d^*\) and its sensitivity to noise strength.

### Scripts
- `scripts/qiskit_qaoa_protocol.py` (Qiskit Aer)
- `scripts/pennylane_qaoa_protocol.py` (PennyLane default.mixed)

## 2. Depth Gap Illustration (Classical vs Quantum)
We provide `scripts/depth_gap_plot.py` to visualize how optimal depth scales in classical local models vs quantum models (fault-tolerant and NISQ-capped).

## 3. Figures and Template
- Run `python scripts/make_figures.py` to regenerate the φ–κ³ overlay, non-Gaussian variant, and a synthetic fitting scaffold.
- A starter CSV (`data/rolt_phi_kappa3_template.csv`) is included for your own stability-vs-depth measurements.

## 4. Caveats
- The φ and κ³ forms are **hypothesis-driven fits**. Some systems may show different profiles.
- Results depend on graph size, noise model, and parameter choices.
- This repository avoids external hardware references; all conclusions are from our own simulations.
