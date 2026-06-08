"""
rolt_aps_simulation_v2.py
=========================

This is a corrective rewrite of the original ROLT/APS toy simulation.

What it fixes:
1. No accidental `eig(...)[0]` bug: the full eigenvalue array is used.
2. No hidden randomness: the default run uses fixed phases.
3. Twisted boundaries now insert an actual boundary bond before phasing it.
4. APS-style eta is only reported for Hermitian spectra, with an explicit
   low-energy window so bulk modes do not dominate the diagnostic.
5. Parameter sweeps report spectral flow proxies through changes in the
   negative-mode count and the minimum gap.

What it does not claim:
1. A genuinely non-Hermitian APS eta. If a matrix is non-Hermitian, this file
   refuses to label the result an APS invariant.
2. A proof of topology from a single eta number. The sweep diagnostics are the
   more honest test in this finite-dimensional toy model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib
import numpy as np
from scipy.linalg import eigh, eigvals

matplotlib.use("Agg")
import matplotlib.pyplot as plt


I_FACT = 0.498015668 - 0.154949828j
DEFAULT_ALPHAS = (0.618034, 0.414214, 0.732051)
DEFAULT_PHASES = (0.0, 0.7, 1.4)


@dataclass
class SpectrumSummary:
    name: str
    hermitian: bool
    hermiticity_error: float
    gap: float
    eigenvalues: np.ndarray
    eta_full: float | None
    eta_low: float | None
    low_mode_count: int
    note: str = ""


def build_rolt_potential(
    L: int,
    lambda_: float,
    r: float,
    alphas: tuple[float, float, float] = DEFAULT_ALPHAS,
    phases: tuple[float, float, float] = DEFAULT_PHASES,
) -> np.ndarray:
    """Three-layer quasi-periodic potential with deterministic phases."""
    weights = np.array([1.0, r, r**2], dtype=float)
    weights /= np.linalg.norm(weights)
    sites = np.arange(L, dtype=float)
    V = np.zeros(L, dtype=float)
    for i in range(3):
        V += lambda_ * weights[i] * np.cos(2.0 * np.pi * alphas[i] * sites + phases[i])
    return V


def build_domain_wall_profile(L: int, m_left: float, m_right: float, width: float = 4.0) -> np.ndarray:
    """Smooth domain-wall profile interpolating from m_left to m_right."""
    x = np.linspace(-1.0, 1.0, L)
    midpoint = 0.5 * (m_left + m_right)
    contrast = 0.5 * (m_right - m_left)
    return midpoint + contrast * np.tanh(width * x)


def hopping_parameters(option: str) -> tuple[complex, complex]:
    """Forward hopping amplitudes used in the original toy model."""
    if option in {"standard", "hermitian"}:
        return -0.5j, 0.5j
    if option in {"non_herm", "gamma_hop"}:
        # Despite the legacy name, this still produces a Hermitian matrix below.
        return -I_FACT / 2.0, I_FACT.conjugate() / 2.0
    if option in {"real_abs", "real_abs_i_fact"}:
        magnitude = abs(I_FACT) / 2.0
        return -magnitude, magnitude
    raise ValueError(f"unknown hopping option: {option}")


def add_spinor_bond(
    D: np.ndarray,
    left_site: int,
    right_site: int,
    hop: complex,
    hop_dag: complex,
    phase: complex = 1.0 + 0.0j,
) -> None:
    """Insert the nearest-neighbour spinor bond used by the original script."""
    li = 2 * left_site
    ri = 2 * right_site

    D[li, ri + 1] += phase * hop
    D[ri + 1, li] += np.conjugate(phase) * hop.conjugate()

    D[li + 1, ri] += phase * hop_dag
    D[ri, li + 1] += np.conjugate(phase) * hop_dag.conjugate()


def build_dirac_matrix(
    L: int,
    V: np.ndarray,
    m0: float = 0.0,
    option: str = "standard",
    twist: float = 0.0,
    boundary: str = "open",
    wilson_strength: float = 0.2,
    mass_profile: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 2L x 2L lattice Dirac operator used in the ROLT toy model."""
    if len(V) != L:
        raise ValueError("V must have length L")
    if mass_profile is not None and len(mass_profile) != L:
        raise ValueError("mass_profile must have length L")

    D = np.zeros((2 * L, 2 * L), dtype=complex)
    hop, hop_dag = hopping_parameters(option)
    onsite_mass = m0 + V if mass_profile is None else m0 + V + mass_profile

    for site, mass in enumerate(onsite_mass):
        idx = 2 * site
        D[idx, idx] = mass + wilson_strength
        D[idx + 1, idx + 1] = -mass + wilson_strength

    for site in range(L - 1):
        add_spinor_bond(D, site, site + 1, hop, hop_dag)

    if boundary == "open":
        return D
    if boundary == "periodic":
        add_spinor_bond(D, L - 1, 0, hop, hop_dag)
        return D
    if boundary == "twisted":
        add_spinor_bond(D, L - 1, 0, hop, hop_dag, phase=np.exp(1j * twist))
        return D
    raise ValueError(f"unknown boundary type: {boundary}")


def hermiticity_error(D: np.ndarray) -> float:
    return float(np.max(np.abs(D - D.conjugate().T)))


def is_hermitian(D: np.ndarray, atol: float = 1e-10) -> bool:
    return hermiticity_error(D) <= atol


def compute_eigenvalues(D: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """Return the eigenvalues, whether the matrix is Hermitian, and the error."""
    error = hermiticity_error(D)
    hermitian = error <= 1e-10
    if hermitian:
        evals = eigh(D, eigvals_only=True)
    else:
        evals = eigvals(D)
    return np.asarray(evals), hermitian, error


def smoothed_eta(evals: np.ndarray, delta: float) -> float:
    """Smoothed sign sum used as the finite-size eta diagnostic."""
    values = np.real(evals)
    return float(np.sum(values / np.sqrt(values**2 + delta**2)))


def compute_eta_windowed(evals: np.ndarray, delta: float, cutoff: float) -> tuple[float, int]:
    """Low-energy eta restricted to |lambda| < cutoff."""
    values = np.real(evals)
    mask = np.abs(values) < cutoff
    if not np.any(mask):
        return 0.0, 0
    eta = np.sum(values[mask] / np.sqrt(values[mask] ** 2 + delta**2))
    return float(eta), int(np.count_nonzero(mask))


def summarize_spectrum(name: str, D: np.ndarray, delta: float, cutoff: float) -> SpectrumSummary:
    evals, hermitian, error = compute_eigenvalues(D)
    gap = float(np.min(np.abs(evals)))
    if hermitian:
        eta_full = smoothed_eta(evals, delta)
        eta_low, count = compute_eta_windowed(evals, delta, cutoff)
        note = ""
    else:
        eta_full = None
        eta_low = None
        count = 0
        note = "non-Hermitian spectrum: APS eta is not reported"
    return SpectrumSummary(
        name=name,
        hermitian=hermitian,
        hermiticity_error=error,
        gap=gap,
        eigenvalues=evals,
        eta_full=eta_full,
        eta_low=eta_low,
        low_mode_count=count,
        note=note,
    )


def negative_mode_count(evals: np.ndarray, tol: float = 1e-9) -> int:
    """Count negative real eigenvalues away from zero for Hermitian families."""
    values = np.real(evals)
    return int(np.count_nonzero(values < -tol))


def scan_hermitian_family(
    parameters: np.ndarray,
    matrix_builder,
    delta: float,
    cutoff: float,
) -> dict[str, np.ndarray | list[tuple[float, float]] | float]:
    """Track gap and negative-mode count across a Hermitian one-parameter family."""
    gaps = []
    negative_counts = []
    low_eta_values = []

    for param in parameters:
        D = matrix_builder(float(param))
        evals, hermitian, error = compute_eigenvalues(D)
        if not hermitian:
            raise ValueError(f"scan requires Hermitian matrices; error={error:.3e} at param={param}")
        gaps.append(float(np.min(np.abs(evals))))
        negative_counts.append(negative_mode_count(evals))
        low_eta, _ = compute_eta_windowed(evals, delta=delta, cutoff=cutoff)
        low_eta_values.append(low_eta)

    gaps = np.asarray(gaps)
    negative_counts = np.asarray(negative_counts)
    low_eta_values = np.asarray(low_eta_values)

    crossings = []
    for i in range(len(parameters) - 1):
        if negative_counts[i + 1] != negative_counts[i]:
            crossings.append((float(parameters[i]), float(parameters[i + 1])))

    min_gap_idx = int(np.argmin(gaps))
    return {
        "parameters": parameters,
        "gaps": gaps,
        "negative_counts": negative_counts,
        "low_eta_values": low_eta_values,
        "net_flow": float(negative_counts[-1] - negative_counts[0]),
        "crossing_windows": crossings,
        "min_gap_parameter": float(parameters[min_gap_idx]),
        "min_gap_value": float(gaps[min_gap_idx]),
    }


def count_sign_changes(profile: np.ndarray, tol: float = 1e-9) -> int:
    """Count sign changes in a real profile, ignoring exact zeros."""
    signs = np.sign(profile)
    last = 0.0
    cleaned = []
    for value in signs:
        if abs(value) <= tol:
            continue
        if last != 0.0 and value != last:
            cleaned.append(1)
        last = value
    return len(cleaned)


def edge_mean(profile: np.ndarray, width: int = 5) -> tuple[float, float]:
    width = max(1, min(width, len(profile) // 2))
    left = float(np.mean(profile[:width]))
    right = float(np.mean(profile[-width:]))
    return left, right


def print_summary(summary: SpectrumSummary, cutoff: float) -> None:
    print(summary.name)
    print(f"  Hermitian: {summary.hermitian}   ||D-D^dagger||_max = {summary.hermiticity_error:.3e}")
    print(f"  Gap min |lambda|: {summary.gap:.6f}")
    if summary.hermitian:
        print(f"  Full-spectrum smoothed sum: {summary.eta_full:+.6f}   [bulk-dominated]")
        print(
            f"  Low-energy eta (|lambda| < {cutoff:.2f}): {summary.eta_low:+.6f}   "
            f"using {summary.low_mode_count} modes"
        )
    else:
        print(f"  {summary.note}")


def plot_full_spectra(summaries: list[SpectrumSummary], cutoff: float, output_path: str) -> None:
    """Save a panel plot of the full spectrum for each case."""
    n_cases = len(summaries)
    ncols = 2
    nrows = math.ceil(n_cases / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    real_scale = max(np.max(np.abs(np.real(summary.eigenvalues))) for summary in summaries)
    imag_scale = max(np.max(np.abs(np.imag(summary.eigenvalues))) for summary in summaries)
    real_scale = max(real_scale, 1.0)
    imag_scale = max(imag_scale, 1e-6)

    low_handle = None
    bulk_handle = None

    for ax, summary in zip(axes, summaries):
        evals = np.asarray(summary.eigenvalues)
        ax.set_title(summary.name, fontsize=10)

        if summary.hermitian:
            values = np.sort(np.real(evals))
            indices = np.arange(len(values))
            mask = np.abs(values) < cutoff

            ax.axhspan(-cutoff, cutoff, color="#f4e7c5", alpha=0.45, lw=0)
            ax.axhline(0.0, color="#333333", lw=1.0)

            bulk_handle = ax.scatter(
                indices[~mask],
                values[~mask],
                s=18,
                color="#1d3557",
                alpha=0.85,
                edgecolors="none",
                label="bulk modes",
            )
            low_handle = ax.scatter(
                indices[mask],
                values[mask],
                s=26,
                color="#c1121f",
                alpha=0.95,
                edgecolors="none",
                label=f"|lambda| < {cutoff:.2f}",
            )

            ax.set_xlabel("sorted eigenvalue index")
            ax.set_ylabel("eigenvalue")
            ax.set_ylim(-1.08 * real_scale, 1.08 * real_scale)

            eta_label = "n/a" if summary.eta_low is None else f"{summary.eta_low:+.3f}"
            ax.text(
                0.02,
                0.96,
                f"gap={summary.gap:.3e}\neta_low={eta_label}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
        else:
            ax.axhline(0.0, color="#333333", lw=1.0)
            ax.axvline(0.0, color="#333333", lw=1.0)
            bulk_handle = ax.scatter(
                np.real(evals),
                np.imag(evals),
                s=20,
                color="#1d3557",
                alpha=0.85,
                edgecolors="none",
                label="spectrum",
            )
            ax.set_xlabel("Re lambda")
            ax.set_ylabel("Im lambda")
            ax.set_xlim(-1.08 * real_scale, 1.08 * real_scale)
            ax.set_ylim(-1.08 * imag_scale, 1.08 * imag_scale)
            ax.text(
                0.02,
                0.96,
                f"gap proxy={summary.gap:.3e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

    for ax in axes[n_cases:]:
        ax.axis("off")

    handles = [handle for handle in (bulk_handle, low_handle) if handle is not None]
    labels = [handle.get_label() for handle in handles]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=len(handles),
            frameon=False,
        )

    fig.suptitle("ROLT Dirac Full Spectra", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    # Core parameters
    L = 40
    lambda_ = 1.2
    r = 0.65
    m0 = 0.1
    delta = 0.01
    cutoff = 0.5

    phases = DEFAULT_PHASES
    V_tri = build_rolt_potential(L, lambda_, r, phases=phases)
    V_single = build_rolt_potential(L, lambda_, r=0.0, phases=phases)

    print("=== ROLT APS Simulation v2 ===")
    print(f"L={L}, lambda={lambda_}, r={r}, m0={m0}, delta={delta}, low-energy cutoff={cutoff}")
    print(f"Fixed phases = {tuple(round(p, 3) for p in phases)}")
    print()

    # Domain wall designed to have opposite asymptotic signs before adding the ROLT potential.
    domain_wall = build_domain_wall_profile(L, m_left=-2.0, m_right=2.0, width=4.0)
    effective_mass = m0 + V_tri + domain_wall
    left_edge, right_edge = edge_mean(effective_mass, width=5)
    sign_changes = count_sign_changes(effective_mass)

    print("Domain-wall diagnostic")
    print(f"  effective mass range: [{effective_mass.min():+.6f}, {effective_mass.max():+.6f}]")
    print(f"  left/right edge means: {left_edge:+.6f}, {right_edge:+.6f}")
    print(f"  discrete sign changes in m_eff(x): {sign_changes}")
    print()

    cases = [
        (
            "1. Gamma-hop tri-layer (legacy 'non_herm' option, but matrix is Hermitian here)",
            build_dirac_matrix(L, V_tri, m0=m0, option="non_herm", boundary="open"),
        ),
        (
            "2. Standard Hermitian tri-layer",
            build_dirac_matrix(L, V_tri, m0=m0, option="standard", boundary="open"),
        ),
        (
            "3. Standard Hermitian domain-wall tri-layer",
            build_dirac_matrix(
                L,
                V_tri,
                m0=m0,
                option="standard",
                boundary="open",
                mass_profile=domain_wall,
            ),
        ),
        (
            "4. Gamma-hop tri-layer with actual twisted periodic boundary",
            build_dirac_matrix(L, V_tri, m0=m0, option="non_herm", boundary="twisted", twist=0.3),
        ),
        (
            "5. Baseline single-layer",
            build_dirac_matrix(L, V_single, m0=m0, option="standard", boundary="open"),
        ),
    ]

    print("Case summaries")
    summaries = []
    for name, D in cases:
        summary = summarize_spectrum(name, D, delta=delta, cutoff=cutoff)
        summaries.append(summary)
        print_summary(summary, cutoff=cutoff)
        print()

    print("Parameter sweeps")

    m0_values = np.linspace(-3.0, 3.0, 121)
    m0_scan = scan_hermitian_family(
        m0_values,
        matrix_builder=lambda mass: build_dirac_matrix(
            L,
            V_tri,
            m0=mass,
            option="standard",
            boundary="open",
        ),
        delta=delta,
        cutoff=cutoff,
    )
    print("  Uniform mass sweep on the tri-layer Hermitian chain")
    print(
        f"    smallest gap = {m0_scan['min_gap_value']:.6f} at m0 = {m0_scan['min_gap_parameter']:+.3f}"
    )
    print(f"    net change in negative-mode count = {m0_scan['net_flow']:+.0f}")
    if m0_scan["crossing_windows"]:
        windows = ", ".join(f"[{a:+.3f}, {b:+.3f}]" for a, b in m0_scan["crossing_windows"])
        print(f"    crossing windows from sign-count jumps: {windows}")
    else:
        print("    no sign-count jump detected on this grid")

    lambda_values = np.linspace(0.0, 2.0, 81)
    lambda_scan = scan_hermitian_family(
        lambda_values,
        matrix_builder=lambda lam: build_dirac_matrix(
            L,
            build_rolt_potential(L, lam, r, phases=phases),
            m0=m0,
            option="standard",
            boundary="open",
        ),
        delta=delta,
        cutoff=cutoff,
    )
    print("  Lambda sweep on the tri-layer Hermitian chain")
    print(
        f"    smallest gap = {lambda_scan['min_gap_value']:.6f} at lambda = {lambda_scan['min_gap_parameter']:+.3f}"
    )
    print(f"    net change in negative-mode count = {lambda_scan['net_flow']:+.0f}")
    if lambda_scan["crossing_windows"]:
        windows = ", ".join(f"[{a:+.3f}, {b:+.3f}]" for a, b in lambda_scan["crossing_windows"])
        print(f"    crossing windows from sign-count jumps: {windows}")
    else:
        print("    no sign-count jump detected on this grid")

    print()
    print("Interpretation")
    print("  The full-spectrum smoothed sum is printed only as a cautionary comparison.")
    print("  The low-energy eta window is the quantity to inspect for zero-mode asymmetry.")
    print("  The mass and lambda sweeps are the cleaner tests for genuine spectral flow.")

    plot_path = "rolt_aps_simulation_v2_full_spectra.png"
    plot_full_spectra(summaries, cutoff=cutoff, output_path=plot_path)
    print(f"  Saved full-spectrum plot to {plot_path}")


if __name__ == "__main__":
    main()
