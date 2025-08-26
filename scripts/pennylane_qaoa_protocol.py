# scripts/pennylane_qaoa_protocol.py
# QAOA(MaxCut) on default.qubit vs default.mixed with depolarizing noise
# Run: python scripts/pennylane_qaoa_protocol.py

import numpy as np, networkx as nx, itertools
import pennylane as qml

def random_3_regular_graph(n, seed=5):
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

def cost_unitary(gamma, G):
    for (u,v) in G.edges():
        qml.CNOT(wires=[u, v])
        qml.RZ(2*gamma, wires=v)
        qml.CNOT(wires=[u, v])

def mixer_unitary(beta, n):
    for q in range(n):
        qml.RX(2*beta, wires=q)

def qaoa_ansatz(params, G):
    n=G.number_of_nodes()
    for q in range(n): qml.Hadamard(wires=q)
    for (gamma,beta) in params:
        cost_unitary(gamma,G); mixer_unitary(beta,n)

def qaoa_ansatz_noisy(params, G, gamma_noise=0.01):
    n=G.number_of_nodes()
    for q in range(n):
        qml.Hadamard(wires=q); qml.DepolarizingChannel(gamma_noise, wires=q)
    for (gamma,beta) in params:
        for (u,v) in G.edges():
            qml.CNOT(wires=[u,v]); qml.DepolarizingChannel(gamma_noise, wires=v)
            qml.RZ(2*gamma, wires=v); qml.DepolarizingChannel(gamma_noise, wires=v)
            qml.CNOT(wires=[u,v]); qml.DepolarizingChannel(gamma_noise, wires=v)
        for q in range(n):
            qml.RX(2*beta, wires=q); qml.DepolarizingChannel(gamma_noise, wires=q)

def default_params(p):
    gammas=np.linspace(0.5,1.2,p); betas=np.linspace(0.4,0.9,p)
    return np.vstack([gammas,betas]).T

def run_experiment(n=8, p_max=10, shots=20000, noise_gamma=0.01, seed=9):
    G=random_3_regular_graph(n, seed=seed); max_cut=max_cut_bruteforce(G)
    dev_ideal = qml.device('default.qubit', wires=G.number_of_nodes(), shots=shots)
    dev_noisy = qml.device('default.mixed', wires=G.number_of_nodes(), shots=shots)

    @qml.qnode(dev_ideal)
    def circuit_ideal(params):
        qaoa_ansatz(params, G); return qml.sample(qml.PauliZ(wires=range(G.number_of_nodes())))

    @qml.qnode(dev_noisy)
    def circuit_noisy(params):
        qaoa_ansatz_noisy(params, G, gamma_noise=noise_gamma)
        return qml.sample(qml.PauliZ(wires=range(G.number_of_nodes())))

    def z_to_bits(sample): return ((1 - sample) // 2).astype(int)

    def hist(samples):
        counts={}
        for row in samples:
            bits=z_to_bits(row)
            bitstr=''.join(str(int(b)) for b in bits)
            counts[bitstr]=counts.get(bitstr,0)+1
        return counts

    d_vals=np.arange(1,p_max+1); approx_ratio=[]; fidelity_proxy=[]
    for p in d_vals:
        params=default_params(p)
        samp_ideal=circuit_ideal(params); samp_noisy=circuit_noisy(params)

        counts_ideal=hist(samp_ideal); counts_noisy=hist(samp_noisy)

        mean_cut=0.0
        for bitstring,c in counts_ideal.items():
            bits=tuple(int(b) for b in bitstring)
            mean_cut += (c/shots)*cut_value(G,bits)
        approx_ratio.append(mean_cut/max_cut)

        keys=set(counts_ideal)|set(counts_noisy)
        pprob=np.array([counts_ideal.get(k,0)/shots for k in keys])
        qprob=np.array([counts_noisy.get(k,0)/shots for k in keys])
        fidelity_proxy.append(np.sum(np.sqrt(pprob*qprob)))

    print("Depths:", list(d_vals))
    print("alpha(d):", [round(x,4) for x in approx_ratio])
    print("F(d):", [round(x,4) for x in fidelity_proxy])

if __name__=='__main__':
    run_experiment()
