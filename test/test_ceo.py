import unittest
import numpy as np
from numerov.ceo import calculate_energy_psi 

class TestNumerovSolution(unittest.TestCase):

    def test_solves_ceo(self):
        nominal_energy_levels = [
            [4.75506982e+01, 1.50000000e+00],
            [3.72094680e+02, 1.57432152e+02],
            [7.23622957e+02, 5.29192129e+02],
            [1.33511059e+03, 1.14469373e+03],
            [2.18043336e+03, 1.99143664e+03],
            [3.22692330e+03, 3.03868677e+03],
            [4.40978607e+03, 4.22215161e+03],
            [5.61686755e+03, 5.43001110e+03],
            [6.68565714e+03, 6.50057814e+03],
            [7.42216871e+03, 7.24636154e+03]
        ]

        # NB: *_ (star in front of variable "_") 
        #    is just an example of enhanced output unpacking in python3
        energy_levels, *_ = calculate_energy_psi(nx0=2)

        # Test only every 100 value
        np.testing.assert_almost_equal(
            energy_levels[1::100],
            nominal_energy_levels,
            decimal=5,
            err_msg="The values of energy leves have changed"
        )


