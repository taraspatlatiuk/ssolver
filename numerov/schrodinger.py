import numpy as np
import scipy.linalg as spla

from schrodinger_plot import output
from schrodinger_plot import finite_well_plot


def harmonic_potential(x):
    # get well force constant and depth
    omega = 1
    D = 0
    pot = (0.5 * (omega**2) * (x**2)) + D
    return pot


def diagonalize_hamiltonian(Hamiltonian):
    return spla.eigh(Hamiltonian)


def main():
    steps = 1000
    # atomic units
    hbar = 1.0
    m = 1.0
    # divide by two so a well from -W to W is of input width
    W = 20
    # set length variable for xvec
    A = W / 2.0
    # create x-vector from -A to A
    xvec = np.linspace(-A, A, steps)
    # get step size
    h = xvec[1] - xvec[0]
    # create the potential from harmonic potential function
    U = harmonic_potential(xvec)
    # create Laplacian via 3-point finite-difference method
    Laplacian = (
        -2.0 * np.diag(np.ones(steps)) +
        np.diag(np.ones(steps - 1), 1) +
        np.diag(np.ones(steps - 1), -1)
    ) / (float)(h**2)

    # create the Hamiltonian
    Hamiltonian = np.zeros((steps, steps))
    [i, j] = np.indices(Hamiltonian.shape)
    Hamiltonian[i == j] = U
    Hamiltonian += (-0.5) * ((hbar**2) / m) * Laplacian
    # diagonalize the Hamiltonian yielding the wavefunctions and energies
    E, V = diagonalize_hamiltonian(Hamiltonian)
    # determine theoretical number of energy levels (n)
    n = 0
    while E[n] < 10:
        n += 1

    # print output
    output(E, n)

    # create plot
    finite_well_plot(E, V, xvec, steps, n, U)
