# Uniaxial tension for a matrix material
import numpy as np, os
from matplotlib import pyplot as plt
from numba import njit
from scipy.optimize import root
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from sympy import Symbol, simplify, lambdify

plt.rc("text", usetex=True)
plt.style.use("myPlots")

@njit(nogil=True)
def S11(l1, Cv11, Cv22):
    l2 = l1**(-0.5)
    I1 = l1**2. + 2./l1
    Ie1 = l1**2/Cv11 + 2.*l2**2/Cv22
    p = ((mu1 * (I1/3.)**(alph1 - 1.) + mu2 * (I1/3.)**(alph2 - 1.)) * l2 + (m1 * (Ie1/3.)**(a1 - 1.) + m2 * (Ie1/3.)**(a2 - 1.)) * l2/Cv22) * l2
    s1 = ((mu1 * (I1/3.)**(alph1 - 1.) + mu2 * (I1/3.)**(alph2 - 1.)) * l1 + (m1 * (Ie1/3.)**(a1 - 1.) + m2 * (Ie1/3.)**(a2 - 1.)) * l1/Cv11) - p/l1
    return s1


@njit(nogil=True)
def Cvdot11(l1, Cv11, Cv22):
    l2 = l1 ** (-0.5)
    Ie1 = l1**2/Cv11 + 2.*l2**2/Cv22
    A2 = (m1 * (Ie1/3.)**(a1 - 1.) + m2 * (Ie1/3.)**(a2 - 1.))
    Iv1 = Cv11 + 2. * Cv22
    Ie1 = l1**2/Cv11 + 2.*l2**2/Cv22
    Cvinv_C_sq = l1**4./Cv11**2 + 2. * l2 **4./Cv22**2
    Ie2 = 1./2 * (Ie1 ** 2. - Cvinv_C_sq)
    J2NEq = (Ie1**2/3. - Ie2) * A2**2.
    # etaK = etaInf + (eta0 - etaInf + K1 * (Iv1 ** (bta1) - 3**(bta1)))/(1. + (K2 * J2NEq)**(bta2))
    etaK =  eta0
    cvd11 = A2/etaK * (l1**2. - 1./3 * Ie1 * Cv11)
    return cvd11


@njit(nogil=True)
def Cvdot22(l1, Cv11, Cv22):
    l2 = l1 ** (-0.5)
    Ie1 = l1**2/Cv11 + 2.*l2**2/Cv22
    A2 = (m1 * (Ie1/3.)**(a1 - 1.) + m2 * (Ie1/3.)**(a2 - 1.))
    Iv1 = Cv11 + 2. * Cv22
    Ie1 = l1**2/Cv11 + 2.*l2**2/Cv22
    Cvinv_C_sq = l1**4./Cv11**2 + 2. * l2 **4./Cv22**2
    Ie2 = 1./2 * (Ie1 ** 2. - Cvinv_C_sq)
    J2NEq = (Ie1**2/3. - Ie2) * A2**2.
    # etaK = etaInf + (eta0 - etaInf + K1 * (Iv1 ** (bta1) - 3**(bta1)))/(1. + (K2 * J2NEq)**(bta2))
    etaK =  eta0
    cvd22 = A2/etaK * (l2**2. - 1./3 * Ie1 * Cv22)
    return cvd22


@njit(nogil=True)
def kterms(dt, Cv_n, l1, l1_n):

    G = np.zeros((6, 2))
    Cv = np.zeros_like(Cv_n)
    Cv11_n, Cv22_n = Cv_n

    G1, G2, G3, G4, G5, G6 = G

    l1_half = l1_n + 0.5 * (l1 - l1_n)

    l1_quarter = l1_n + 0.25 * (l1 - l1_n)

    l1_three_quarter = l1_n + 0.75 * (l1 - l1_n)

    G1[0] = Cvdot11(l1_n, Cv11_n, Cv22_n)
    G1[1] = Cvdot22(l1_n, Cv11_n, Cv22_n)

    G2[0] = Cvdot11(
        l1_half,
        Cv11_n + G1[0] * dt / 2.0,
        Cv22_n + G1[1] * dt / 2.0,)
    G2[1] = Cvdot22(
        l1_half,
        Cv11_n + G1[0] * dt / 2.0,
        Cv22_n + G1[1] * dt / 2.0,)

    G3[0] = Cvdot11(
        l1_quarter,
        Cv11_n + (3.0 * G1[0] + G2[0]) * dt / 16.0,
        Cv22_n + (3.0 * G1[1] + G2[1]) * dt / 16.0,)
    G3[1] = Cvdot22(
        l1_quarter,
        Cv11_n + (3.0 * G1[0] + G2[0]) * dt / 16.0,
        Cv22_n + (3.0 * G1[1] + G2[1]) * dt / 16.0,)

    G4[0] = Cvdot11(
        l1_half,
        Cv11_n + G3[0] * dt / 2.0,
        Cv22_n + G3[1] * dt / 2.0,)
    G4[1] = Cvdot22(
        l1_half,
        Cv11_n + G3[0] * dt / 2.0,
        Cv22_n + G3[1] * dt / 2.0,)

    G5[0] = Cvdot11(
        l1_three_quarter,
        Cv11_n + 3.0 * dt / 16.0 * (-G2[0] + 2.0 * G3[0] + 3.0 * G4[0]),
        Cv22_n + 3.0 * dt / 16.0 * (-G2[1] + 2.0 * G3[1] + 3.0 * G4[1]),)
    G5[1] = Cvdot22(
        l1_three_quarter,
        Cv11_n + 3.0 * dt / 16.0 * (-G2[0] + 2.0 * G3[0] + 3.0 * G4[0]),
        Cv22_n + 3.0 * dt / 16.0 * (-G2[1] + 2.0 * G3[1] + 3.0 * G4[1]),)

    G6[0] = Cvdot11(
        l1,
        Cv11_n
        + dt / 7.0 * (G1[0] + 4.0 * G2[0] + 6.0 * G3[0] - 12.0 * G4[0] + 8.0 * G5[0]),
        Cv22_n
        + dt / 7.0 * (G1[1] + 4.0 * G2[1] + 6.0 * G3[1] - 12.0 * G4[1] + 8.0 * G5[1]),)
    G6[1] = Cvdot22(
        l1,
        Cv11_n
        + dt / 7.0 * (G1[0] + 4.0 * G2[0] + 6.0 * G3[0] - 12.0 * G4[0] + 8.0 * G5[0]),
        Cv22_n
        + dt / 7.0 * (G1[1] + 4.0 * G2[1] + 6.0 * G3[1] - 12.0 * G4[1] + 8.0 * G5[1]),)

    Cv = Cv_n + dt / 90.0 * (7.0 * G1 + 32.0 * G3 + 12.0 * G4 + 32.0 * G5 + 7.0 * G6)
    return Cv


# load_type = "sine"
load_type = "triangle"

if load_type == "triangle":
    ldot = 1.
    dt = 0.05
    final_stretch = 2.
    Tf = 2. * (final_stretch - 1.)/ldot
    num_steps = Tf/dt
    timevals = np.linspace(0., Tf, int(num_steps + 1))
    dt_obtained = timevals[1:] - timevals[:-1]

    t1 = timevals[timevals <=Tf/2.]
    t2 = timevals[np.logical_and(timevals > Tf/2., timevals <= Tf)]

    l1vals = np.hstack((
        np.linspace(1., final_stretch, int(len(t1))),
        np.linspace(final_stretch - ldot * dt, 1., int(len(t2))),
    ))

numerical_results_dir = os.path.join(
    os.getcwd(), "results_RK52", f"dt_{dt}", f"{load_type}"
)
os.makedirs(numerical_results_dir, exist_ok=True)

plot_stretches = True
if plot_stretches:
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.plot(timevals, l1vals, label=r"$\lambda_1(t)$")
    ax.set_xlabel(r"Time $(t)$", fontsize=22)
    ax.set_ylabel(r"Stretch: $\lambda(t)$", fontsize=22)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.tick_params(axis="both", pad=10)
    ax.grid(which="major", linestyle="--")
    fig.tight_layout()
    fig.savefig(
        os.path.join(numerical_results_dir, f"stretch_{load_type}.png"),
        dpi=600,
    )
    fig.savefig(
        os.path.join(numerical_results_dir, f"stretch_{load_type}.eps")
    )
    plt.show()

# material parameters:
kap_by_mu = 1e3
mu1 = (0.5) * 0.01
mu2 = (0.5) * 0.01
alph1 = (1.)
alph2 = (1.)
m1 = (0.5)
m2 = (0.5)
a1 = (1.)
a2 = (1.)
K1 = (3507 * 10**3)
K2 = (10**(-6))
bta1 = (1.852)
bta2 = (0.26)
eta0 = (1e2)
etaInf = (0.1)  # 0.1

# time stepping
simulation_already_run = False
if not (simulation_already_run):
    Cv_vals = np.ones((timevals.shape[0], 2))
    S11_vals = np.zeros(timevals.shape[0])
    S22_vals = np.zeros(timevals.shape[0])
    Cv_trial = np.ones(2)
    # G = np.zeros((6, 2))
    max_stag_iters = 5
    print(l1vals[0])
    print(f"S11 at zero = {S11(l1vals[0], Cv_vals[0, 0], Cv_vals[0, 1])}")

    for idx_time, t in enumerate(timevals[1:], start=1):
        Cv_trial = kterms(
            dt,
            Cv_vals[idx_time - 1, :],
            l1vals[idx_time],
            l1vals[idx_time - 1],
        )
        Cv_vals[idx_time] = np.copy(Cv_trial)
        S11_vals[idx_time] = S11(
            l1vals[idx_time],
            Cv_vals[idx_time, 0],
            Cv_vals[idx_time, 1],
        )

    detCv_save = Cv_vals[:, 0] * Cv_vals[:, 1]**2
    print(f"detCv = {detCv_save}")
    # np.savetxt(os.path.join(numerical_results_dir, "detCv_RK5_analytical.txt"), Cv_vals[:, 0] * Cv_vals[:, 1]**2)
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.plot(
        l1vals,
        S11_vals,
        "-bo",
        label=r"FE"
    )
    ax.set_xlabel(r"${\lambda}$", fontsize=22)
    ax.set_ylabel(r"${S}_{11}/\mu_\texttt{m}$", fontsize=24)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.tick_params(axis="both", pad=10)
    ax.grid(which="major", linestyle="--")
    ax.legend(loc=0, ncol=1, fancybox=False, edgecolor="k", fontsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join(numerical_results_dir, "matrix_RK5.png"), dpi=600)
    fig.savefig(os.path.join(numerical_results_dir, "matrix_RK5.eps"))
    plt.close()

    pd.DataFrame(np.vstack((l1vals, S11_vals)).T).to_excel(
        os.path.join(numerical_results_dir, "S11_vs_l11_matrix_RK5.xlsx"),
        index=None,
        header=None,
    )
    Cv_save = np.zeros((timevals.shape[0], 3, 3))

    for i in range(Cv_save.shape[0]):
        Cv_save[i] = np.array([
            [Cv_vals[i, 0], 0, 0],
            [0, Cv_vals[i, 1], 0],
            [0., 0., Cv_vals[i, 1]]
        ])
    np.savez_compressed(os.path.join(numerical_results_dir, "Cv_values_Analytical.npz"), Cvvalues=Cv_save, detCvvalues=detCv_save)
else:
    l1vals, S11_vals = (
        pd.read_excel(
            os.path.join(numerical_results_dir, "S11_vs_l11_matrix_RK5.xlsx"),
            index_col=None,
            header=None,
        )
        .to_numpy()
        .T
    )
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.plot(
        l1vals, S11_vals / mu_m, "r-", label=r"Approximation"
    )  #: $\overline{S}_{11}(\overline{\bf F}, \overline{\bf C}^v)$")
    ax.set_xlabel(r"${\lambda}$", fontsize=24)
    ax.set_ylabel(r"${S}_{11}/\mu_\texttt{m}$", fontsize=24)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", pad=10)
    ax.axhline(0.0, linestyle="--", color="k", lw=0.8)  # horizontal lines
    ax.axvline(1.0, linestyle="--", color="k", lw=0.8)  # vertical lines
    ax.legend(loc=0, ncol=1, fancybox=False, edgecolor="k", fontsize=24)
    fig.tight_layout()
    fig.savefig(
        os.path.join(numerical_results_dir, "matrix_RK5.png"),
        dpi=600,
    )
    fig.savefig(os.path.join(numerical_results_dir, "matrix_RK5.eps"))
    plt.close()

    detCv_be = np.loadtxt(os.path.join(numerical_results_dir, "detCv_be.txt"))
    detCv_RK5 = np.loadtxt(os.path.join(numerical_results_dir, "detCv_RK5.txt"))
    fig, ax = plt.subplots(1,1,figsize=(9, 8))
    ax.plot(l1vals, detCv_RK5, "r", label="RK5")
    ax.annotate(f"$\Delta t = {dt}$", (l1vals[-1] * 1.01, 0.99998 * detCv_be[-1]), fontsize=22)
    ax.plot(l1vals, detCv_be, "-bo", label="Backward Euler", markevery=5)
    ax.set_xlabel(r"${\lambda}$", fontsize=24)
    ax.set_ylabel(r"$\det~{\bf C}^v$", fontsize=24)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", pad=10)
    ax.grid(which="major", linestyle="--")
    ax.legend(loc=0, ncol=2, fancybox=False, shadow=False, fontsize=22, facecolor="w", edgecolor="w", bbox_to_anchor=(0.1, 1.01), bbox_transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(
        os.path.join(numerical_results_dir, "detCvComparisons.png"),
        dpi=600,
    )
    fig.savefig(os.path.join(numerical_results_dir, "detCvComparisons.eps"))
    plt.show()
