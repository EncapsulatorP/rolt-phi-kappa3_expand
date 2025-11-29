from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_depth_curves(
    n: np.ndarray,
    alpha_local: float,
    alpha_resid: float,
    beta_quantum: float,
    noise_lambda: float,
) -> dict:
    """Return simple depth scaling curves for classical/quantum regimes."""
    d_class_local = alpha_local * n
    d_class_resid = alpha_resid * np.log2(n)
    d_quantum_ft = beta_quantum * np.log2(n)
    d_noise_cap = np.full_like(n, 1.0 / max(noise_lambda, 1e-9))
    d_quantum_nisq = np.minimum(d_quantum_ft, d_noise_cap)
    return {
        "Classical O(n)": d_class_local,
        "Classical residual O(log n)": d_class_resid,
        "Quantum FT O(log n)": d_quantum_ft,
        "Quantum NISQ capped": d_quantum_nisq,
    }


def plot_depth_gap(n: np.ndarray, curves: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, values in curves.items():
        ax.plot(n, values, label=label, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Depth (layers)")
    ax.set_title("Depth gap: classical vs quantum scaling (toy illustration)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot toy depth-scaling gaps for classical vs quantum regimes.")
    parser.add_argument("--n_min", type=float, default=1e3, help="Minimum problem size.")
    parser.add_argument("--n_max", type=float, default=1e5, help="Maximum problem size.")
    parser.add_argument("--points", type=int, default=200, help="Number of sample points.")
    parser.add_argument("--alpha_local", type=float, default=0.02, help="Slope for O(n) classical depth.")
    parser.add_argument("--alpha_resid", type=float, default=3.0, help="Scale for residual/log-depth classical model.")
    parser.add_argument("--beta_quantum", type=float, default=1.2, help="Scale for fault-tolerant quantum log-depth.")
    parser.add_argument("--noise_lambda", type=float, default=0.02, help="Depolarizing-like cap for NISQ depth.")
    parser.add_argument("--figs_dir", type=Path, default=Path("figs"), help="Directory for plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = np.logspace(np.log10(args.n_min), np.log10(args.n_max), args.points)
    curves = compute_depth_curves(
        n=n,
        alpha_local=args.alpha_local,
        alpha_resid=args.alpha_resid,
        beta_quantum=args.beta_quantum,
        noise_lambda=args.noise_lambda,
    )
    fig_path = args.figs_dir / "depth_gap.png"
    plot_depth_gap(n, curves, fig_path)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
