from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def gaussian_bump(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Simple log-concave bump used for coherence/stability toy curves."""
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def generate_profiles(
    d_min: float,
    d_max: float,
    num_points: int,
    phi_mode: float,
    kappa_mode: float,
    sigma_phi: float,
    sigma_kappa: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    depths = np.linspace(d_min, d_max, num_points)
    s_phi = gaussian_bump(depths, phi_mode, sigma_phi)
    s_kappa = gaussian_bump(depths, kappa_mode, sigma_kappa)
    ell = s_phi * s_kappa
    d_star = float(depths[np.argmax(ell)])
    return depths, s_phi, s_kappa, d_star


def save_template_csv(depths: np.ndarray, ell: np.ndarray, path: Path, stride: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.column_stack((depths[::stride], ell[::stride]))
    header = "depth,ell"
    np.savetxt(path, payload, fmt="%.4f", delimiter=",", header=header, comments="")


def plot_profiles(
    depths: np.ndarray,
    s_phi: np.ndarray,
    s_kappa: np.ndarray,
    ell: np.ndarray,
    d_star: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(depths, s_phi, label="S_phi(d): coherence", linewidth=2)
    ax.plot(depths, s_kappa, label="S_kappa(d): stability", linewidth=2)
    ax.plot(depths, ell, label="ell(d) = S_phi * S_kappa", linewidth=2, color="#444")
    ax.axvline(d_star, linestyle=":", color="red", label=f"d* = {d_star:.2f}")
    ax.set_xlabel("Depth (d)")
    ax.set_ylabel("Normalized scale")
    ax.set_title("ROLT-phi-kappa3 toy profile")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROLT-phi-kappa3 toy figures and template CSV.")
    parser.add_argument("--phi2", type=float, default=((1 + np.sqrt(5)) / 2) ** 2, help="Mode for S_phi(d).")
    parser.add_argument("--kappa_mode", type=float, default=3.5, help="Mode for S_kappa(d).")
    parser.add_argument("--sigma_phi", type=float, default=0.6, help="Width of S_phi(d).")
    parser.add_argument("--sigma_kappa", type=float, default=1.0, help="Width of S_kappa(d).")
    parser.add_argument("--d_min", type=float, default=0.5, help="Minimum depth to plot.")
    parser.add_argument("--d_max", type=float, default=8.0, help="Maximum depth to plot.")
    parser.add_argument("--points", type=int, default=400, help="Number of depth samples.")
    parser.add_argument("--stride", type=int, default=20, help="Stride when writing the template CSV.")
    parser.add_argument("--figs_dir", type=Path, default=Path("figs"), help="Directory for generated figures.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Directory for generated data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depths, s_phi, s_kappa, d_star = generate_profiles(
        d_min=args.d_min,
        d_max=args.d_max,
        num_points=args.points,
        phi_mode=args.phi2,
        kappa_mode=args.kappa_mode,
        sigma_phi=args.sigma_phi,
        sigma_kappa=args.sigma_kappa,
    )
    ell = s_phi * s_kappa

    fig_path = args.figs_dir / "rolt_phi_kappa3_fig1.png"
    csv_path = args.data_dir / "rolt_phi_kappa3_template.csv"

    plot_profiles(depths, s_phi, s_kappa, ell, d_star, fig_path)
    save_template_csv(depths, ell, csv_path, stride=args.stride)

    print(f"Saved {fig_path} (d* = {d_star:.2f})")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
