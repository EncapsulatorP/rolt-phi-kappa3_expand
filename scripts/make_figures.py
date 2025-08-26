import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
out = Path('figs'); out.mkdir(exist_ok=True)

phi = (1+np.sqrt(5))/2; phi2 = phi**2
mu_k = 3.5; sigma_phi = 0.6; sigma_k = 1.0
d = np.linspace(0.5, 8.0, 400)

S_phi = np.exp(-((d-phi2)**2)/(2*sigma_phi**2))
S_k = np.exp(-((d-mu_k)**2)/(2*sigma_k**2))
ell = S_phi*S_k
d_star = d[np.argmax(ell)]

plt.plot(d,S_phi,label='S_phi'); plt.plot(d,S_k,label='S_kappa')
plt.plot(d,ell,label='ell(d)'); plt.axvline(d_star,ls=':',label=f'd*={d_star:.2f}')
plt.legend(); plt.savefig('figs/rolt_phi_kappa3_fig1.png'); plt.close()

rows=['d,stability']+[f"{di:.2f},{yi:.4f}" for di,yi in zip(d[::20],ell[::20])]
Path('data/rolt_phi_kappa3_template.csv').write_text("\n".join(rows))
print('Generated figs and template CSV')
