import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
#from numerov.timer import Timer
from timer import Timer

# NB: That's how one can separate physical model from
# parameters of the algorithm
class CeoPotential:
    def __init__(self, omega=1, d=0, mag=30.0, width=2.0, hbar=1, mass=1):
        self.omega = omega
        self.width = width
        self.mag = mag
        self.d = d

        # NB: All units should be members of Potentai classes
        self.hbar = hbar
        self.mass = mass

    def _ceo_dopping(self, x, x0):
        ceo_pot = self.mag / self.width * x - self.mag * (1 + x0 / self.width)
        ceo_pot[(ceo_pot > 0) | (ceo_pot < -self.mag)] = 0
        return ceo_pot

    def _harmonic(self, x, x0):
        return 0.5 * self.omega**2 * (x - x0)**2 + self.d

    def potential(self, x, x0):
        return self._harmonic(x, x0)# + self._ceo_dopping(x, -10)


def diagonalize_hamiltonian(Hamiltonian):
    return spla.eigh(Hamiltonian)

class Solver:
    def __init__(self, task, size, stepsize):
        self.task = task
        A = self.ndiag(-2.0, size) / stepsize ** 2
        B = self.ndiag(10.0, size) / 12.0
        # Free hamiltonian term in numerov method
        # $ H_{\text{free}} = -\frac{\hbar^{2}}{2m}B^{-1}A \psi $
        self.free_hamiltonian = -0.5 * task.hbar ** 2 / task.mass * spla.inv(B).dot(A)

    @staticmethod
    def ndiag(coef, size):
        # Create diagonal matrix for 3-point finite-difference method
        return (coef
            * np.diag(np.ones(size))
            + np.diag(np.ones(size - 1), 1)
            + np.diag(np.ones(size - 1), -1)
        )

    def _evaluate(self, xvec, x0):
        # evaluate the potential
        U = self.task.potential(xvec, x0)

        # evaluate the Hamiltonian
        # $$ -\frac{\hbar^{2}}{2m}B^{-1}A \psi + U \psi = E \psi$$
        hamiltonian = self.free_hamiltonian + np.diag(U)

        # diagonalize the Hamiltonian yielding the wavefunctions and energies
        # TODO: Why do we call it diagonalize?
        E, V = diagonalize_hamiltonian(hamiltonian)
        return E, V, U

    def evaluate_x0(self, nx0, x0vec, nsteps, xvec):
        Eall = np.zeros((nsteps, nx0))
        psiall = np.zeros((nsteps, nsteps, nx0))
        # TODO: use meaningful names here
        for i, x0 in enumerate(x0vec):
            Evec,Vvec,Uvec = self._evaluate(xvec, x0)
            psiall[:,:,i] = Vvec[:,:]**2
            Eall[:,i] = Evec[:]
        return Eall, psiall, xvec, x0vec

def calculate_energy_psi(x0, nsteps=1000, x_min=-10., x_max=10.):
    # to calculate energies and wave functions for one specific x0
    dx = (x_max - x_min)/nsteps
    xvec = np.linspace(x_min+dx, x_max-dx, nsteps-1)
    #print(xvec)

    with Timer("Computation time"):
        solver = Solver(CeoPotential(), nsteps-1, dx)
        eall, psiall, xvec = solver._evaluate(xvec, x0)
    return eall, psiall, xvec

def calculate_energy_psi_x0(nx0=40, x0_min=-20., x0_max=20., nsteps=1000, x_min=-10., x_max=10.):
    # to calculate energies and wave functions for all x0 in the range x0_min to x0_max
    dx = (x_max - x_min)/nsteps
    xvec = np.linspace(x_min+dx, x_max-dx, nsteps-1)
    x0vec = np.linspace(x0_min, x0_max, nx0)
    with Timer("Computation time"):
        solver = Solver(CeoPotential(), nsteps-1, dx)
        eall, psiall, xvec, x0vec = solver.evaluate_x0(nx0, x0vec, nsteps-1, xvec)
    return eall, psiall, xvec, x0vec

def calculate_density(eall, psiall, E_fermi):
    # eall: index 0 - mode number, index 1 - x0 value
    # psiall: index 0 - x coordinate, index 1 - mode number, index 2 - x0 value
    mask_3D = np.zeros(psiall.shape, dtype=bool)
    mask_3D[:,:,:] = eall[np.newaxis,:,:]>E_fermi
    psiall_f = np.copy(psiall)
    psiall_f[mask_3D] = 0
    density = np.sum(psiall_f,axis=(2))
    total_density = np.sum(density,axis=(1))
    return density, total_density
