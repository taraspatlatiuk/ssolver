from ceo import calculate_energy_psi
from ceo import calculate_energy_psi_x0
from ceo import calculate_density
from Schrodinger_plot import plot_E
from Schrodinger_plot import plot_1D
from Schrodinger_plot import plot_2D

import numpy as np

def main():
    """
    # TODO: Move all parameters except for constants here
    Eall, psiall, xvec = calculate_energy_psi(x0=0.)
    print(
        "Energy levels: {}".format(Eall[10:13]),
        # "Wave functions?: {}".format(psiall),
        # "Positions???: {}".format(xvec),
        # "Initial conditions?: {}".format(x0vec)
    )

    Eall, psiall, xvec = calculate_energy_psi(x0=-10.)
    print(
        "Energy levels: {}".format(Eall[10:13])
        )

    Eall, psiall, xvec = calculate_energy_psi(x0=10.)
    print(
        "Energy levels: {}".format(Eall[10:13])
        )
    """

    Eall, psiall, xvec, x0vec = calculate_energy_psi_x0(nx0=200)
    density, total_density = calculate_density(Eall, psiall, E_fermi=5)

    plot_1D(xvec, total_density)
    plot_1D(xvec, density[:,0])

if __name__ == '__main__':
    main()
