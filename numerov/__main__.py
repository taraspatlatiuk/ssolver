from ceo import calculate_energy_psi
from ceo import calculate_energy_psi_x0
from Schrodinger_plot import plot_E


def main():
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

    Eall, psiall, xvec, x0vec = calculate_energy_psi_x0(nx0=10)
    plot_E(x0vec,Eall,30)

if __name__ == '__main__':
    main()
