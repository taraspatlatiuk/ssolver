import unittest
import numpy as np
from numerov.ceo import calculate_energy_psi

class TestNumerovSolution(unittest.TestCase):

    def test_solves_ceo(self):
        bulk_energy_levels = 0.5 + np.linspace(0, 30, 31)
        energy_levels, *_ = calculate_energy_psi(x0=0.)

        # Test only every 100 value
        np.testing.assert_almost_equal(
            energy_levels[0:31],
            bulk_energy_levels,
            decimal=3,
            err_msg="The values of energy leves have changed"
        )

    def test_left_edge(self):
        exect_energy_levels = 0.5 + np.linspace(1, 61, 31)
        energy_levels, *_ = calculate_energy_psi(x0=-10.)

        # Test only every 100 value
        np.testing.assert_almost_equal(
            exect_energy_levels[0:31],
            energy_levels,
            decimal=1,
            err_msg="The values of energy leves have changed"
        )

    def test_right_edge(self):
        exect_energy_levels = 0.5 + np.linspace(1, 61, 31)
        energy_levels, *_ = calculate_energy_psi(x0=10.)
        #print(energy_levels[0:31])
        # Test only every 100 value
        np.testing.assert_almost_equal(
            exect_energy_levels[0:31],
            energy_levels,
            decimal=1,
            err_msg="The values of energy leves have changed"
        )
