# Next Steps

Priorities for extending the ROLT-phi-kappa3 simulations.

---

## Immediate / Low-effort

- **Variational QAOA angles.** The current QAOA sweep uses fixed heuristic parameters. Optimize gamma/beta with COBYLA or BFGS to expose the true coherence/stability trade-off and locate the depth optimum d* empirically.
- **QAOA noise sweep.** Run `qaoa_maxcut_qiskit_aer.py` over a range of `--noise-lambda` values (e.g. 0.001, 0.005, 0.01, 0.05) to map how d* shifts as a function of noise strength.
- **Save QAOA output to CSV.** Both QAOA scripts print to stdout; add a `--out` argument writing to `outputs/data/` for downstream analysis.
- **Fix random phases in `rolt_aps_eta_3case.py`.** The original script draws phases randomly, making results irreproducible. Adopt the fixed-phase convention from the v2 script.

---

## Near-term Physics

- **IPR and Lyapunov exponent sweeps.** The paper describes (but does not yet implement) sweeps of the Inverse Participation Ratio and the transfer-matrix Lyapunov exponent over (lambda, r) at fixed variance. These are the primary falsifiable signatures of the phi-kappa3 tri-layer effect.
- **Multifractality at the critical strip.** Near lambda ≈ 2, compute the generalized fractal dimension D_q = tau(q)/(q-1) for the tri-layer vs single-layer AA model at matched variance. The phi-kappa3 prediction is D_q(tri-layer) < D_q(single-layer).
- **Mobility edge in the domain-wall case.** The domain-wall case (v2 script, case 3) is the only one with negative low-energy eta (-1.00), suggesting a topological boundary mode. Confirm this is a genuine in-gap state by resolving the eigenvalue spectrum near zero vs system size L.

---

## Medium-term Extensions

- **Zeteon frequency selector.** The paper describes selecting the three driving frequencies alpha_1, alpha_2, alpha_3 via the Diophantine score S_L(alpha) applied to the Zeteon-derived family Z(p). Implement this selector and compare against the current fixed alphas (0.618, 0.414, 0.732).
- **2D Harper / Hofstadter model.** Extend the tri-layer quasi-periodic driving to 2D: one axis driven by Zeteon-selected phi, one axis under a variance budget. Compute Chern numbers and Hall conductance.
- **Entropy budget enforcement.** The weight vector w = (1, r, r^2) / ||w|| enforces fixed total variance. Add a script that sweeps r at fixed ||w|| to verify the tri-layer effect is not a trivial variance artifact.
- **Replace ad-hoc layers with continued-fraction convergents.** As flagged in the paper's critique of GCF numbers: derive the three layers canonically from the convergents of alpha*, and compare kappa-allocation schemes systematically.

---

## Infrastructure

- **Shared graph/model fixtures.** The QAOA and Dirac scripts each define their own random graph and potential builders. Extract common utilities to a `src/rolt/` module to avoid drift.
- **Parameter sweep runner.** Write a thin sweep harness that calls any script over a grid of CLI arguments and collects stdout CSV rows into a single `outputs/data/` file.
- **Figures pipeline.** Add a top-level `Makefile` or `run_all.py` that regenerates all outputs in dependency order.
