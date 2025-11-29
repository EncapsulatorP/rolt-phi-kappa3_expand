# Methods: Simulation Protocols for ROLT-phi-kappa3

This note summarizes the simulations backing the ROLT-phi-kappa3 framing. Results are illustrative and limited to our controlled runs.

## 1. QAOA-style quantum circuits (MaxCut, 3-regular graphs)

**Task**: MaxCut on connected 3-regular graphs with even `n` (default `n=8` for brute-force reference).  
**Depth**: QAOA layer count `d = p`.  
**Metrics**:  
- Noiseless approximation ratio `alpha(d)`: expected cut value under the ideal distribution divided by the true optimum.  
- Noisy fidelity proxy `F(d)`: Hellinger affinity between ideal and noisy bitstring histograms.  
**Models**:  
- `S_phi(d)`: coherence/gain bump near `phi^2`.  
- `S_kappa(d)`: stability/capacity decay with mode near 3-4 (Gaussian or `sech^2` style).  

**Protocol**
1. Draw a connected 3-regular graph (reject until connected).
2. Sweep depths `d = 1..p_max`.
3. Evaluate `alpha(d)` with an ideal simulator; evaluate `F(d)` with a depolarizing-noise simulator.
4. Fit or visualize `S_phi`, `S_kappa`, and their product `ell(d)`.
5. Identify the maximizer `d*` and its sensitivity to noise strength.

**Scripts**
- `scripts/qiskit_qaoa_protocol.py` (Qiskit + Aer noise model)
- `scripts/pennylane_qaoa_protocol.py` (PennyLane default.qubit vs default.mixed)

## 2. Depth gap illustration (classical vs quantum)
`scripts/depth_gap_plot.py` visualizes how optimal depth scaling differs across classical local models, residual/log-depth models, fault-tolerant quantum regimes, and NISQ-capped quantum regimes.

## 3. Figures and data template
`python scripts/make_figures.py` regenerates the coherence/stability/product overlay and writes `data/rolt_phi_kappa3_template.csv` for your own stability-vs-depth measurements.

## 4. Caveats
- ROLT-phi-kappa3 is a hypothesis-driven fit, not a theorem.
- Results depend on graph size, noise model, and parameter choices.
- We report only our own simulations; no external hardware data are included.
