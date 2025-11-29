"""QAOA(MaxCut) sweep using PennyLane (ideal vs depolarizing noise).

Run:
    python scripts/pennylane_qaoa_protocol.py
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import networkx as nx
import numpy as np
import pennylane as qml


@dataclass
class QAOAResult:
    depths: np.ndarray
    approx_ratio: np.ndarray
    fidelity_proxy: np.ndarray


def random_3_regular_graph(n: int, seed: int, max_tries: int = 50) -> nx.Graph:
    if n % 2 or n < 4:
        raise ValueError("n must be even and at least 4 for a 3-regular graph.")
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        candidate = nx.random_regular_graph(3, n, seed=int(rng.integers(1, 1_000_000_000)))
        if nx.is_connected(candidate):
            return candidate
    raise RuntimeError(f"Failed to sample a connected 3-regular graph after {max_tries} attempts.")


def cut_value(G: nx.Graph, bitstring: Sequence[int]) -> int:
    return sum(1 for u, v in G.edges() if bitstring[u] != bitstring[v])


def max_cut_bruteforce(G: nx.Graph) -> int:
    n = G.number_of_nodes()
    best = 0
    for bits in itertools.product([0, 1], repeat=n):
        best = max(best, cut_value(G, bits))
    return best


def cost_unitary(gamma: float, G: nx.Graph) -> None:
    for (u, v) in G.edges():
        qml.CNOT(wires=[u, v])
        qml.RZ(2 * gamma, wires=v)
        qml.CNOT(wires=[u, v])


def mixer_unitary(beta: float, n: int) -> None:
    for q in range(n):
        qml.RX(2 * beta, wires=q)


def qaoa_ansatz(params: Sequence[Tuple[float, float]], G: nx.Graph) -> None:
    n = G.number_of_nodes()
    for q in range(n):
        qml.Hadamard(wires=q)
    for gamma, beta in params:
        cost_unitary(gamma, G)
        mixer_unitary(beta, n)


def qaoa_ansatz_noisy(params: Sequence[Tuple[float, float]], G: nx.Graph, gamma_noise: float) -> None:
    n = G.number_of_nodes()
    for q in range(n):
        qml.Hadamard(wires=q)
        qml.DepolarizingChannel(gamma_noise, wires=q)
    for gamma, beta in params:
        for (u, v) in G.edges():
            qml.CNOT(wires=[u, v])
            qml.DepolarizingChannel(gamma_noise, wires=v)
            qml.RZ(2 * gamma, wires=v)
            qml.DepolarizingChannel(gamma_noise, wires=v)
            qml.CNOT(wires=[u, v])
            qml.DepolarizingChannel(gamma_noise, wires=v)
        for q in range(n):
            qml.RX(2 * beta, wires=q)
            qml.DepolarizingChannel(gamma_noise, wires=q)


def default_params(p: int) -> np.ndarray:
    gammas = np.linspace(0.5, 1.2, p)
    betas = np.linspace(0.4, 0.9, p)
    return np.vstack([gammas, betas]).T


def counts_from_samples(samples: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in samples:
        bitstr = "".join(str(int(b)) for b in row)
        counts[bitstr] = counts.get(bitstr, 0) + 1
    return counts


def hellinger_affinity(counts_p: Dict[str, int], counts_q: Dict[str, int], shots: int) -> float:
    keys = set(counts_p) | set(counts_q)
    return float(
        sum(np.sqrt(counts_p.get(k, 0) / shots * counts_q.get(k, 0) / shots) for k in keys)
    )


def run_experiment(
    n: int = 8,
    p_max: int = 10,
    shots: int = 5000,
    noise_gamma: float = 0.01,
    seed: int = 9,
) -> QAOAResult:
    if p_max < 1:
        raise ValueError("p_max must be >= 1.")

    G = random_3_regular_graph(n, seed=seed)
    max_cut = max_cut_bruteforce(G)
    num_wires = G.number_of_nodes()

    dev_ideal = qml.device("default.qubit", wires=num_wires, shots=shots)
    dev_noisy = qml.device("default.mixed", wires=num_wires, shots=shots)

    @qml.qnode(dev_ideal)
    def circuit_ideal(params):
        qaoa_ansatz(params, G)
        return qml.sample(wires=range(num_wires))

    @qml.qnode(dev_noisy)
    def circuit_noisy(params):
        qaoa_ansatz_noisy(params, G, gamma_noise=noise_gamma)
        return qml.sample(wires=range(num_wires))

    depths = np.arange(1, p_max + 1)
    approx_ratio = []
    fidelity_proxy = []

    for p in depths:
        params = default_params(p)
        samp_ideal = circuit_ideal(params)
        samp_noisy = circuit_noisy(params)

        counts_ideal = counts_from_samples(samp_ideal)
        counts_noisy = counts_from_samples(samp_noisy)

        mean_cut = 0.0
        for bitstring, c in counts_ideal.items():
            bits = tuple(int(b) for b in bitstring)
            mean_cut += (c / shots) * cut_value(G, bits)
        approx_ratio.append(mean_cut / max_cut)

        fidelity_proxy.append(hellinger_affinity(counts_ideal, counts_noisy, shots))

    return QAOAResult(
        depths=np.array(depths, dtype=int),
        approx_ratio=np.array(approx_ratio, dtype=float),
        fidelity_proxy=np.array(fidelity_proxy, dtype=float),
    )


def format_results(result: QAOAResult) -> str:
    lines = ["depth,alpha,F"]
    for d, a, f in zip(result.depths, result.approx_ratio, result.fidelity_proxy):
        lines.append(f"{d},{a:.4f},{f:.4f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAOA MaxCut sweep with PennyLane (ideal vs noisy).")
    parser.add_argument("--n", type=int, default=8, help="Number of nodes (even, >=4).")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum QAOA depth p.")
    parser.add_argument("--shots", type=int, default=5000, help="Number of shots per depth.")
    parser.add_argument("--noise-gamma", type=float, default=0.01, help="Depolarizing noise strength.")
    parser.add_argument("--seed", type=int, default=9, help="RNG seed for graph sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        n=args.n, p_max=args.p_max, shots=args.shots, noise_gamma=args.noise_gamma, seed=args.seed
    )
    print(format_results(result))


if __name__ == "__main__":
    main()
