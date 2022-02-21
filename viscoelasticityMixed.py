# finite viscoelasticity: mixed formulation (Taylor-Hood)
import numpy as np, os, meshio
from scipy.sparse import bmat
from typing import Tuple
from skfem.helpers import grad, transpose, det, inv, identity
from skfem import (
    MeshTet, BilinearForm, LinearForm, solve, condense, ElementVectorH1, ElementTetDG,
    ElementTetCCR, ElementTetP1, Basis, Functional, asm, project, ElementTetP2, MeshTet2,
    DiscreteField
)
import numpy.linalg as nla
from skfem.io import from_meshio
import matplotlib.pyplot as plt
import pandas as pd
from pypardiso import spsolve as pyspsolve


plt.rc("text", usetex=True)
plt.style.use("myPlots")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
# material parameters
mu, lmbda = 1., 1.e3
nu, eta = 1., 1.

def doubleInner(A, B):
    return np.einsum("ij...,ij...", A, B)

def F1(w):
    u = w["disp"]
    p = w["press"]
    Cv = w["Cv"]
    Cvinv = inv(Cv.value)
    F = grad(u) + identity(u)
    Finv = inv(F)
    J = det(F)
    FCvinv = np.einsum("ik...,kj...->ij...", F, Cvinv)
    return mu * F + p * J * transpose(Finv) + nu * FCvinv

def sPiola(w):
    u = w["disp"]
    p = w["press"]
    Cv = w["Cv"]
    Cvinv = inv(Cv)
    F = grad(u) + identity(u)
    Finv = inv(F)
    J = det(F)
    FCvinv = np.einsum("ik...,kj...->ij...", F, Cvinv)
    return mu * F + p * J * transpose(Finv) + nu * FCvinv

def sPiola33(w):
    return F1(w)[2, 2]

def F2(w):
    u = w["disp"]
    p = w["press"].value
    F = grad(u) + identity(u)
    J = det(F)
    Js = .5 * (lmbda + p + 2. * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2)) / lmbda
    dJsdp = ((.25 * lmbda + .25 * p + .5 * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2))
             / (lmbda * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2)))
    return J - (Js + (p + mu / Js - lmbda * (Js - 1)) * dJsdp)

def A11(w):
    u = w["disp"]
    p = w["press"]
    Cv = w["Cv"]
    eye = identity(u)
    F = grad(u) + eye
    J = det(F)
    Cvinv = inv(Cv.value)

    Finv = inv(F)
    L = (p * J * np.einsum("lk...,ji...->ijkl...", Finv, Finv)
         - p * J * np.einsum("jk...,li...->ijkl...", Finv, Finv)
         + mu * np.einsum("ik...,jl...->ijkl...", eye, eye) + 
         + nu * np.einsum("ik...,lj...->ijkl...", eye, Cvinv))
    return L

def A12(w):
    u = w["disp"]
    F = grad(u) + identity(u)
    J = det(F)
    Finv = inv(F)
    return J * transpose(Finv)

def A22(w):
    u = w["disp"]
    p = w["press"].value
    Js = .5 * (lmbda + p + 2. * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2)) / lmbda
    dJsdp = ((.25 * lmbda + .25 * p + .5 * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2))
             / (lmbda * np.sqrt(lmbda * mu + .25 * (lmbda + p) ** 2)))
    d2Jdp2 = .25 * mu / (lmbda * mu + .25 * (lmbda + p) ** 2) ** (3/2)
    L = (-2. * dJsdp - p * d2Jdp2 + mu / Js ** 2 * dJsdp ** 2 - mu / Js * d2Jdp2
         + lmbda * (Js - 1.) * d2Jdp2 + lmbda * dJsdp ** 2)
    return L

def volume(w):
    dw = w["disp"].grad
    F = dw + identity(dw)
    J = det(F)
    return J

def Cvdot(Cv, C, nu_m, eta_m):
    Cvinv = inv(Cv)
    CCvinv = doubleInner(C, Cvinv)
    Cvdot = nu_m/eta_m * (C - 1./3 * CCvinv * Cv)
    return Cvdot

def calculateCStrain(gradu):
    F = gradu + identity(gradu)
    C = np.einsum("ki...,kj...->ij...", F, F)
    return C

def evolEq(dt, Cvn, C, Cn, nu_m, eta_m):
    C_half = Cn + 0.5 * (C - Cn)
    C_quart = Cn + 0.25 * (C - Cn)
    C_three_quart = Cn + 0.75 * (C - Cn)
    G1 = Cvdot(Cvn, Cn, nu_m, eta_m)
    G2 = Cvdot(Cvn + G1 * dt/2., C_half, nu_m, eta_m)
    G3 = Cvdot(Cvn + dt/16. * (3. * G1 + G2), C_quart, nu_m, eta_m)
    G4 = Cvdot(Cvn + G3 * dt/2., C_half, nu_m, eta_m)
    G5 = Cvdot(Cvn + 3. * dt/16. * (-G2 + 2. * G3 + 3. * G4), C_three_quart, nu_m, eta_m)
    G6 = Cvdot(Cvn + dt/7. * (G1 + 4. * G2 + 6. * G3 - 12. * G4 + 8. * G5), C, nu_m, eta_m)
    return Cvn + dt / 90.0 * (7.0 * G1 + 32.0 * G3 + 12.0 * G4 + 32.0 * G5 + 7.0 * G6)


def solveSystem(du, dp, Cv, I, basis, itr, solver="pypardiso"):
    uv = basis["u"].interpolate(du)
    pv = basis["p"].interpolate(dp)
    K11 = asm(b11, basis["u"], basis["u"], disp=uv, press = pv, Cv=Cv)
    K12 = asm(b12, basis["p"], basis["u"], disp=uv, press = pv)
    K22 = asm(b22, basis["p"], basis["p"], disp=uv, press = pv)
    f = np.concatenate((
        asm(a1, basis["u"], disp=uv, press = pv, Cv=Cv),
        asm(a2, basis["p"], disp=uv, press = pv)
    ))
    K = bmat(
        [[K11, K12],
         [K12.T, K22]], "csr"
    )
    # uvp = solve(*condense(K, -f, I=I))
    if solver != "pypardiso":
        uvp = solve(*condense(K, -f, I=I))
    else:
        A, b, x, I = condense(K, -f, I=I)
        uvp = x.copy()
        uvp[I] = pyspsolve(A, b)
    delu, delp = np.split(uvp, [du.shape[0]])
    du += delu
    dp += delp
    normu = nla.norm(delu)
    normp = nla.norm(delp)
    norm_res = nla.norm(f[I])
    print(f"Newton iter: {itr+1}, norm_du: {normu:.8e}, norm_dp: {normp:.8e}, norm_res = {norm_res:.8e}")
    sbar = stress.assemble(basis["u"], disp=uv, press=pv, Cv=Cv)
    Cstrain = calculateCStrain(uv.grad)
    return (norm_res, np.copy(du), np.copy(dp), sbar, Cstrain)
    
def convertCRToTet(u:np.ndarray, p:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uNew = project(u, basis_from=basis["u"], basis_to=project_basis["u"])
    pNew = project(p, basis_from=basis["p"], basis_to=project_basis["p"])
    return (uNew, pNew)

def stitchu(u, basis):
    nodal_disps = u[basis["u"].nodal_dofs].T
    edge_disps = u[basis["u"].edge_dofs].T

    pointData = np.concatenate((
    nodal_disps, edge_disps), axis=0)
    return pointData


# mesh = from_meshio(meshio.xdmf.read("fenicsmesh.xdmf"))
# mesh = from_meshio(generateMesh())
mesh = MeshTet()
# uelem = ElementVectorH1(ElementTetP2())
# pelem = ElementTetP1()
uelem = ElementVectorH1(ElementTetCCR())
pelem = ElementTetDG(ElementTetP1())
elems = {
    "u": uelem,
    "p": pelem
}

elems_proj = {
    "u": ElementVectorH1(ElementTetP2()),
    "p": ElementTetP1()
}
basis = {
    field: Basis(mesh, e, intorder=4)
    for field, e in elems.items()
}
project_basis = {
    field: Basis(mesh, e, intorder=4)
    for field, e in elems_proj.items()
}

du = np.zeros(basis["u"].N)
dp = -1. * (mu + nu) * np.ones(basis["p"].N)

stretch_ = 0.0

dofsold = [
    basis["u"].find_dofs({"left":mesh.facets_satisfying(lambda x: x[0] < 1.e-6)}, skip=["u^2", "u^3"]),
    basis["u"].find_dofs({"bottom":mesh.facets_satisfying(lambda x: x[1] < 1.e-6)}, skip=["u^1", "u^3"]),
    basis["u"].find_dofs({"back":mesh.facets_satisfying(lambda x: x[2] < 1.e-6)}, skip=["u^1", "u^2"]),
    basis["u"].find_dofs({"front":mesh.facets_satisfying(lambda x: np.abs(x[2]-1.) < 1.e-6 )}, skip=["u^1", "u^2"])
]

dofs = {}
for dof in dofsold:
    dofs.update(dof)

du[dofs["left"].all()] = 0.
du[dofs["bottom"].all()] = 0.
du[dofs["back"].all()] = 0.
du[dofs["front"].all()] = stretch_

I = np.hstack((
    basis["u"].complement_dofs(dofs),
    basis["u"].N + np.arange(basis["p"].N)
))

@LinearForm(nthreads=3)
def a1(v, w):
    return np.einsum("ij...,ij...", F1(w), grad(v))

@LinearForm(nthreads=3)
def a2(v, w):
    return F2(w) * v

@BilinearForm(nthreads=3)
def b11(u, v, w):
    return np.einsum("ijkl...,ij...,kl...", A11(w), grad(u), grad(v))

@BilinearForm(nthreads=3)
def b12(u, v, w):
    return np.einsum("ij...,ij...", A12(w), grad(v)) * u

@BilinearForm(nthreads=3)
def b22(u, v, w):
    return A22(w) * u * v

@Functional(nthreads=3)
def vol(w):
    return volume(w)

@Functional(nthreads=3)
def stress(w):
    return sPiola33(w)


ldot = 1.
dt = 0.05
final_stretch = 2.
Tf = 2. * (final_stretch - 1.)/ldot
num_steps = Tf/dt
timevals = np.linspace(0., Tf, int(num_steps + 1))
dt_obtained = timevals[1:] - timevals[:-1]

t1 = timevals[timevals <=Tf/2.]
t2 = timevals[np.logical_and(timevals > Tf/2., timevals <= Tf)]

stretchVals = np.hstack((
    np.linspace(1., final_stretch, int(len(t1))),
    np.linspace(final_stretch - ldot * dt, 1., int(len(t2))),
))

meshioPoints = mesh.p.T
meshioCells = [("tetra", mesh.t.T)]
# meshioCells = [("tetra", mesh.t.T)]
results_dir = os.path.join(os.getcwd(), "resultsViscoTimeSteps_no_mu")
os.makedirs(results_dir, exist_ok=True)
filename = os.path.join(results_dir, f"disps_{ldot}.xdmf")

avgStress = np.zeros_like(stretchVals)

dispVal = stretchVals[0] - 1.
du[dofs["front"].all()] = dispVal
duv = basis["u"].interpolate(du)
pv = basis["p"].interpolate(dp).value
Cv = identity(duv.grad)
Cvn = Cv.copy()
Cn = Cv.copy()
Cstrain = Cv.copy()
# np.testing.assert_allclose(Cv, np.eye(3)[:, :, np.newaxis, np.newaxis].repeat(Cv.shape[-2], axis=-2).repeat(Cv.shape[-1], axis=-1))
# np.testing.assert_allclose(Cvn, np.eye(3)[:, :, np.newaxis, np.newaxis].repeat(Cv.shape[-2], axis=-2).repeat(Cv.shape[-1], axis=-1))
# np.testing.assert_allclose(Cn, np.eye(3)[:, :, np.newaxis, np.newaxis].repeat(Cv.shape[-2], axis=-2).repeat(Cv.shape[-1], axis=-1))


print(du[basis["u"].nodal_dofs].T.shape)
print(mesh)

with meshio.xdmf.TimeSeriesWriter(filename) as writer:
    writer.write_points_cells(meshioPoints, meshioCells)
    for idx, tv in enumerate(timevals):
        print(f"Time: {tv} of {Tf}")
        dispVal = stretchVals[idx] - 1.
        du[dofs["front"].all()] = dispVal
        maxiters = 10
        max_stag_iters = 3
        for itr in range(maxiters):
            norm_res, du, dp, sbar, Cstrain = solveSystem(du, dp, DiscreteField(Cv), I, basis, itr)
            # print(f"res: {norm_res}")
            Cv = evolEq(dt, Cvn, Cstrain, Cn, nu, eta) # do one more pass
            norm_res, du, dp, sbar, Cstrain = solveSystem(du, dp, DiscreteField(Cv), I, basis, itr)
            # print(f"res2: {norm_res2}")
            if norm_res < 1.e-8:
                converged = True
                break
        Cvn = Cv.copy()
        Cn = Cstrain.copy()
        np.testing.assert_allclose(det(Cvn), 1.)
        vol_def = vol.assemble(basis["u"], disp=basis["u"].interpolate(du))
        print(f"deformed_volume: {vol_def}, res: {norm_res}")
        du_projected = project(du, basis_from=basis["u"], basis_to=project_basis["u"])
        writer.write_data(
            tv, point_data={
                "u": du_projected[project_basis["u"].nodal_dofs].T,
            }, cell_data={
                "p": [dp[basis["p"].interior_dofs].T.mean(axis=1)]
            }
        )
        avgStress[idx] = sbar

np.savetxt(os.path.join(results_dir, f"stresses_{ldot}.txt"), avgStress)
np.savetxt(os.path.join(results_dir, f"stretches_{ldot}.txt"), stretchVals)
np.savetxt(os.path.join(results_dir, f"timevals_{ldot}.txt"), timevals)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(stretchVals, avgStress, "r-", label=r"FEM")
ax.set_xlabel("$\lambda$", fontsize=22)
ax.set_ylabel("$\dfrac{S}{\mu}$", fontsize=22, rotation=0, labelpad=12)
ax.legend(loc=0, ncol=2, fontsize=22, fancybox=False, edgecolor="k")
ax.grid(which="major", linestyle="--")
fig.tight_layout()
fig.savefig(os.path.join(results_dir, f"Svl_{ldot}.png"), dpi=600)
plt.show()


# final_vol = vol.assemble(basis["u"], disp=uv)
# final_stress = sPiola33(grad(uv), pv.value) #pointwise
# final_stress_averaged = stress.assemble(basis["u"], disp=uv, press=pv)
# print(f"final stress averaged: {final_stress_averaged}")
# print(final_stress)
# print(f"final stress averaged: {final_stress_averaged}")
# print(final_stress, final_stress_averaged)