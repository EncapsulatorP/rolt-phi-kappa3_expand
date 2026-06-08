import numpy as np
from scipy.linalg import eigh, eigvals

def build_rolt_potential(L, lambda_, r, alphas=None, phases=None):
    if alphas is None:
        alphas = [0.618034, 0.414214, 0.732051]  
    if phases is None:
        phases = [0.0, 0.0, 0.0]
    
    w = np.array([1.0, r, r**2])
    w /= np.sqrt(np.sum(w**2))
    
    n = np.arange(L)
    V = np.zeros(L, dtype=complex)
    for i in range(3):
        V += lambda_ * w[i] * np.cos(2 * np.pi * alphas[i] * n + phases[i])
    return V

def build_dirac_matrix(L, V, m0=0.1, option='standard', wilson=True):
    N = 2 * L
    D = np.zeros((N, N), dtype=complex)
    
    sigma_x = np.array([[0,1],[1,0]], dtype=complex)
    sigma_z = np.array([[1,0],[0,-1]], dtype=complex)
    
    for i in range(L):
        mass = m0 + V[i]
        D[2*i:2*i+2, 2*i:2*i+2] += mass * sigma_z
    
    if option == 'standard':
        hop = 1j / 2.0
    elif option == 'non_herm_i_fact':
        hop = (0.498015668274046 - 0.154949828301810j) / 2.0
    elif option == 'real_abs_i_fact':
        hop = 0.521 / 2.0
    else:  
        hop = 1j / 2.0
    
    for i in range(L-1):
        D[2*i:2*i+2, 2*(i+1):2*(i+1)+2] += hop * sigma_x
        if option == 'non_herm_i_fact':
            D[2*(i+1):2*(i+1)+2, 2*i:2*i+2] += -np.conj(hop) * sigma_x
        else:
            D[2*(i+1):2*(i+1)+2, 2*i:2*i+2] += np.conj(hop) * sigma_x
    
    if wilson:
        for i in range(L):
            D[2*i:2*i+2, 2*i:2*i+2] += 1.0 * sigma_z  
        for i in range(L-1):
            D[2*i:2*i+2, 2*(i+1):2*(i+1)+2] -= 0.5 * sigma_z
            D[2*(i+1):2*(i+1)+2, 2*i:2*i+2] -= 0.5 * sigma_z
    
    return D

def compute_eta(D, delta=0.01):
    if np.iscomplexobj(D) and not np.allclose(D, D.conj().T):
        evals = eigvals(D)
        evals = np.sort_complex(evals)
    else:
        evals = eigh(D)[0]
    eta = np.sum(evals / np.sqrt(evals**2 + delta**2))
    gap = np.min(np.abs(evals))
    return eta, gap

L = 50
lambda_ = 1.2
r = 0.65
m0 = 0.1

V = build_rolt_potential(L, lambda_, r)

options = ['standard', 'non_herm_i_fact', 'real_abs_i_fact', 'single_layer']

results = {}
for opt in options:
    if opt == 'single_layer':
        V_sl = np.zeros(L, dtype=complex)
        D = build_dirac_matrix(L, V_sl, m0, 'standard')
    else:
        D = build_dirac_matrix(L, V, m0, opt)
    
    eta, gap = compute_eta(D)
    results[opt] = {'eta': float(np.real(eta) if np.iscomplexobj(eta) else eta), 'gap': float(gap)}
    print(f"Option {opt}: eta = {eta:.4f}, gap = {gap:.4f}")

print("\nSimulation complete.")
for opt, res in results.items():
    print(opt, res)
