# scripts/qiskit_qaoa_protocol.py
# QAOA(MaxCut, 3-regular graphs): noiseless α(d) and noisy fidelity proxy F(d)
# Run: python scripts/qiskit_qaoa_protocol.py

import numpy as np, networkx as nx, itertools, warnings
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, CommutativeCancellation

def random_3_regular_graph(n, seed=7):
    if n%2 or n<4: raise ValueError("n must be even and ≥4")
    rng=np.random.default_rng(seed)
    for _ in range(20):
        G=nx.random_regular_graph(3, n, seed=int(rng.integers(1,1e9)))
        if nx.is_connected(G): return G
    return G

def cut_value(G, bitstring):
    return sum(1 for u,v in G.edges() if bitstring[u]!=bitstring[v])

def max_cut_bruteforce(G):
    n=G.number_of_nodes(); best=0
    for bits in itertools.product([0,1], repeat=n):
        best=max(best, cut_value(G,bits))
    return best

def qaoa_layer(qc, G, gamma, beta):
    for (u,v) in G.edges():
        qc.cx(u,v); qc.rz(2*gamma,v); qc.cx(u,v)
    for q in range(G.number_of_nodes()):
        qc.rx(2*beta,q)

def qaoa_circuit(G, p, params):
    n=G.number_of_nodes(); qc=QuantumCircuit(n)
    for q in range(n): qc.h(q)
    for k in range(p):
        gamma,beta = params[k]; qaoa_layer(qc,G,gamma,beta)
    qc.measure_all(); return qc

def default_params(p):
    gammas=np.linspace(0.5,1.2,p); betas=np.linspace(0.4,0.9,p)
    return np.vstack([gammas,betas]).T

def build_noise_model(lam=0.01):
    nm=NoiseModel(); dep1=depolarizing_error(lam,1); dep2=depolarizing_error(lam,2)
    for g in ['rz','rx','ry','u','u3','h']: nm.add_all_qubit_quantum_error(dep1, g)
    for g in ['cx','cz','rzz']: nm.add_all_qubit_quantum_error(dep2, g)
    return nm

def hellinger_affinity(counts_p, counts_q, shots):
    keys=set(counts_p)|set(counts_q)
    return sum(np.sqrt(counts_p.get(k,0)/shots * counts_q.get(k,0)/shots) for k in keys)

def run_experiment(n=8, p_max=10, shots=20000, noise_lambda=0.01, seed=7):
    G=random_3_regular_graph(n, seed=seed); max_cut=max_cut_bruteforce(G)
    ideal = AerSimulator(method="statevector")
    noisy = AerSimulator(method="automatic", noise_model=build_noise_model(noise_lambda))

    pm = PassManager([Optimize1qGatesDecomposition(), CommutativeCancellation()])
    d_vals=np.arange(1,p_max+1); approx_ratio=[]; fidelity_proxy=[]

    for p in d_vals:
        params=default_params(p)
        qc=qaoa_circuit(G,p,params); qc=pm.run(qc)

        result_ideal = ideal.run(qc, shots=shots).result()
        counts_ideal = result_ideal.get_counts()

        result_noisy = noisy.run(qc, shots=shots).result()
        counts_noisy = result_noisy.get_counts()

        # Noiseless approximation ratio
        mean_cut=0.0
        for bitstring,c in counts_ideal.items():
            bits=tuple(int(b) for b in bitstring[::-1])  # reverse endian
            mean_cut += (c/shots)*cut_value(G,bits)
        approx_ratio.append(mean_cut/max_cut)

        # Fidelity proxy: Hellinger affinity noisy vs noiseless histograms
        fidelity_proxy.append(hellinger_affinity(counts_ideal, counts_noisy, shots))

    print("Depths:", list(d_vals))
    print("alpha(d):", [round(x,4) for x in approx_ratio])
    print("F(d):", [round(x,4) for x in fidelity_proxy])

if __name__=='__main__':
    run_experiment()
