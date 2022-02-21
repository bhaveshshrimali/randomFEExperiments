from matplotlib import pyplot as plt
import numpy as np, os
from pandas import read_excel

plt.rc("text", usetex=True)
plt.style.use("myPlots")

ode_results = os.path.join(os.getcwd(), r"results_RK5", r"dt_0.05", r"triangle")
fe_results = os.path.join(os.getcwd(), "resultsViscoTimeSteps_no_mu")
Svldata = read_excel(os.path.join(ode_results, r"S11_vs_l11_matrix_RK5.xlsx"), index_col=None, header=None).to_numpy()
l11data = np.loadtxt(os.path.join(fe_results, r"stretches_1.0.txt"))
s11 = np.loadtxt(os.path.join(fe_results, r"stresses_1.0.txt"))

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(Svldata[:,0], Svldata[:,1], 'r-', label='ODE')
ax.plot(l11data, s11, 'bo', label='FE')
ax.set_xlabel(r'$\lambda_{11}$', fontsize=22)
ax.set_ylabel(r'$S_{11}$', fontsize=22)
ax.set_title(r'$S_{11}$ vs $\lambda_{11}$', fontsize=22)
ax.legend(loc='best', fontsize=22)
fig.tight_layout()
fig.savefig("compa.png")
plt.show()