"""QAOA(MaxCut) sweep using Qiskit + Aer noise model.

Run:
    python scripts/qiskit_qaoa_protocol.py
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation, Optimize1qGates


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


def qaoa_layer(qc: QuantumCircuit, G: nx.Graph, gamma: float, beta: float) -> None:
    for (u, v) in G.edges():
        qc.cx(u, v)
        qc.rz(2 * gamma, v)
        qc.cx(u, v)
    for q in range(G.number_of_nodes()):
        qc.rx(2 * beta, q)


def qaoa_circuit(G: nx.Graph, p: int, params: Sequence[Tuple[float, float]]) -> QuantumCircuit:
    qc = QuantumCircuit(G.number_of_nodes())
    qc.h(range(G.number_of_nodes()))
    for gamma, beta in params:
        qaoa_layer(qc, G, gamma, beta)
    qc.measure_all()
    return qc


def default_params(p: int) -> np.ndarray:
    gammas = np.linspace(0.5, 1.2, p)
    betas = np.linspace(0.4, 0.9, p)
    return np.vstack([gammas, betas]).T


def build_noise_model(lam: float = 0.01) -> NoiseModel:
    nm = NoiseModel()
    dep1 = depolarizing_error(lam, 1)
    dep2 = depolarizing_error(lam, 2)
    for gate in ["rz", "rx", "ry", "u", "u3", "h"]:
        nm.add_all_qubit_quantum_error(dep1, gate)
    for gate in ["cx", "cz", "rzz"]:
        nm.add_all_qubit_quantum_error(dep2, gate)
    return nm


def hellinger_affinity(counts_p: Dict[str, int], counts_q: Dict[str, int], shots: int) -> float:
    keys = set(counts_p) | set(counts_q)
    return float(
        sum(np.sqrt(counts_p.get(k, 0) / shots * counts_q.get(k, 0) / shots) for k in keys)
    )


def expectation_from_counts(G: nx.Graph, counts: Dict[str, int], shots: int) -> float:
    mean_cut = 0.0
    for bitstring, c in counts.items():
        bits = tuple(int(b) for b in bitstring[::-1])  # reverse endian to match node indexing
        mean_cut += (c / shots) * cut_value(G, bits)
    return mean_cut


def run_experiment(
    n: int = 8,
    p_max: int = 10,
    shots: int = 8000,
    noise_lambda: float = 0.01,
    seed: int = 7,
) -> QAOAResult:
    if p_max < 1:
        raise ValueError("p_max must be >= 1.")

    G = random_3_regular_graph(n, seed=seed)
    max_cut = max_cut_bruteforce(G)

    ideal = AerSimulator(method="statevector")
    noisy = AerSimulator(method="automatic", noise_model=build_noise_model(noise_lambda))

    pm = PassManager([Optimize1qGates(), CommutativeCancellation()])
    depths = np.arange(1, p_max + 1)
    approx_ratio = []
    fidelity_proxy = []

    for p in depths:
        params = default_params(p)
        qc = qaoa_circuit(G, p, params)
        qc = pm.run(qc)

        result_ideal = ideal.run(qc, shots=shots).result()
        counts_ideal = result_ideal.get_counts()

        result_noisy = noisy.run(qc, shots=shots).result()
        counts_noisy = result_noisy.get_counts()

        mean_cut = expectation_from_counts(G, counts_ideal, shots)
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
    parser = argparse.ArgumentParser(description="QAOA MaxCut sweep with Qiskit + Aer noise model.")
    parser.add_argument("--n", type=int, default=8, help="Number of nodes (even, >=4).")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum QAOA depth p.")
    parser.add_argument("--shots", type=int, default=8000, help="Number of shots per depth.")
    parser.add_argument("--noise-lambda", type=float, default=0.01, help="Depolarizing noise strength.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for graph sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        n=args.n, p_max=args.p_max, shots=args.shots, noise_lambda=args.noise_lambda, seed=args.seed
    )
    print(format_results(result))


if __name__ == "__main__":
    main()
