import numpy as np
from scipy.linalg import eigh, eig
import matplotlib.pyplot as plt

def gamma_function(z):
    # Approximate Gamma(1 + i) 
    return 0.498015668 - 0.154949828j

i_fact = gamma_function(1j)

def build_rolt_potential(L, lambda_, r, alphas=[0.618034, 0.414214, 0.732051], phases=None):
    if phases is None:
        phases = np.random.uniform(0, 2*np.pi, 3)
    w = np.array([1, r, r**2])
    w /= np.sqrt(np.sum(w**2))
    n = np.arange(L)
    V = np.zeros(L, dtype=complex)
    for i in range(3):
        V += lambda_ * w[i] * np.cos(2 * np.pi * alphas[i] * n + phases[i])
    return V.real

def build_dirac_matrix(L, V, m0=0.0, option='standard', twist=0.0):
    N = 2 * L
    D = np.zeros((N, N), dtype=complex)
    
    # Mass term sigma_z ⊗ (m0 + V)
    for i in range(L):
        idx = 2*i
        D[idx, idx] = m0 + V[i]      # upper component
        D[idx+1, idx+1] = -(m0 + V[i])  # lower component
    
    # Hopping term sigma_x ⊗ P
    for i in range(L-1):
        idx1 = 2*i
        idx2 = 2*(i+1)
        
        if option == 'non_herm':
            hop = -i_fact / 2.0
            hop_dag = i_fact.conjugate() / 2.0
        elif option == 'real_abs':
            hop = -abs(i_fact) / 2.0
            hop_dag = abs(i_fact) / 2.0
        else:  # standard hermitian
            hop = -1j / 2.0
            hop_dag = 1j / 2.0
        
        # sigma_x connects upper to lower
        D[idx1, idx2+1] = hop
        D[idx2+1, idx1] = hop.conjugate()
        
        D[idx1+1, idx2] = hop_dag
        D[idx2, idx1+1] = hop_dag.conjugate()
    
    # Wilson term approximation (simple diagonal stabilizer)
    for i in range(L):
        idx = 2*i
        D[idx, idx] += 0.2
        D[idx+1, idx+1] += 0.2
    
    # Twisted boundaries
    if abs(twist) > 1e-8:
        hop_twist = np.exp(1j * twist)
        # Apply to ends
        D[0, 2*(L-1)+1] *= hop_twist
        D[2*(L-1)+1, 0] *= hop_twist.conjugate()
    
    return D

def compute_eta(D, delta=0.01):
    if np.iscomplexobj(D):
        vals = eig(D, right=False)
        evals = vals[0]
    else:
        evals = eigh(D)[0]
    eta = np.sum(evals / np.sqrt(evals**2 + delta**2))
    return eta, np.min(np.abs(evals))

# Parameters
L = 40
lambda_ = 1.2
r = 0.65
m0 = 0.1

V = build_rolt_potential(L, lambda_, r)

print("=== Simulation Results ===")

# 1. Non-Hermitian i! + Tri-layer
D1 = build_dirac_matrix(L, V, m0, option='non_herm')
eta1, gap1 = compute_eta(D1)
print(f"1. Non-Herm i! Tri-layer: eta = {eta1:.4f}, gap = {gap1:.4f}")

# 2. Hermitian Domain-Wall Tri-layer
# Domain wall: m0 varies
V_dw = V + 0.5 * np.tanh(np.linspace(-5,5,L))
D2 = build_dirac_matrix(L, V_dw, m0=0.0, option='standard')
eta2, gap2 = compute_eta(D2)
print(f"2. Hermitian Domain-Wall Tri-layer: eta = {eta2:.4f}, gap = {gap2:.4f}")

# 3. Hybrid Non-Herm + Twisted
D3 = build_dirac_matrix(L, V, m0, option='non_herm', twist=0.3)
eta3, gap3 = compute_eta(D3)
print(f"3. Hybrid Non-Herm + Twist: eta = {eta3:.4f}, gap = {gap3:.4f}")

# Baseline single layer
V_single = build_rolt_potential(L, lambda_, r=0.0)
D_base = build_dirac_matrix(L, V_single, m0, option='standard')
eta_base, gap_base = compute_eta(D_base)
print(f"Baseline Single-layer: eta = {eta_base:.4f}, gap = {gap_base:.4f}")
