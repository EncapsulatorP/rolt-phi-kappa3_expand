import numpy as np, matplotlib.pyplot as plt
n = np.logspace(3,5,200); log2=np.log2
alpha_local=0.02; alpha_resid=3.0; beta_q=1.2; lambda_noise=0.02
d_class_local=alpha_local*n; d_class_resid=alpha_resid*log2(n)
d_quant_ft=beta_q*log2(n); d_noise_cap=int(1/lambda_noise)
d_quant_nisq=np.minimum(d_quant_ft,d_noise_cap)
plt.plot(n,d_class_local,label='Classical O(n)')
plt.plot(n,d_class_resid,label='Classical residual O(log n)')
plt.plot(n,d_quant_ft,label='Quantum FT O(log n)')
plt.plot(n,d_quant_nisq,label='Quantum NISQ capped')
plt.xscale('log'); plt.legend(); plt.savefig('figs/depth_gap.png'); plt.close()
