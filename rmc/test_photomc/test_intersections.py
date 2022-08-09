
import unittest
import numpy as np
import photon_mc_numba_fractional as ph

class TestIntersections(unittest.TestCase):


    def test_get_block(self):

        nx = 10
        ny = 10
        i = 0
        j = 0
        nb = 3
        # x = np.linspace(0.0, 1.0, nx)
        # y = np.linspace(0.0, 1.0, ny)
        x = np.arange(nx, dtype=np.float_)
        y = np.arange(ny, dtype=np.float_)
        # Z = np.zeros((nx, ny))
        Z = np.arange(nx*ny).reshape((nx, ny))
        Y, X = np.meshgrid(y, x)
        # Z = X + Y
        xi, yj, Zij = ph.get_block(i, j, nb, x, y, Z)
        # print(xi)
        # print(yj)
        # print(Zij)
        np.testing.assert_array_almost_equal(xi, np.array([0.0, 1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(yj, np.array([0.0, 1.0, 2.0, 3.0]))


    def test_block_is_near_enough(self):

        # S0 = np.array([0.0, 0.0, 0.0])
        S0 = np.array([5.0, 5.0, 5.0])
        S1 = np.array([1.0, 1.0, 1.0])
        nx = 123
        ny = 123
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(0.0, 1.0, ny)
        Z = np.ones((nx, ny))
        res = ph.is_block_near_enough(S0, S1, x, y, Z)
        self.assertIs(res, True)
        pass


    def test_adaptive(self):
        """
        Test the function to compute intersection points between photon
        ans a surface
        """
        nx = 1203
        ny = 1203
        x = np.linspace(-20, 20, nx)
        y = np.linspace(-20, 20, ny)
        # Z = np.zeros((nx, ny)) + 0.1
        zscale = 1
        Z = np.random.rand(nx*ny).reshape((nx, ny))*zscale + 0.1

        nshoots = 1
        photon = ph.generate_photon(x, y, hTOA=1.0, cosz0=-1, phi0=0)
        photon.S0 = np.array([0.0, 0.0,  2.0])
        l = 100

        for iss in range(nshoots):
            photon.cosz = -np.random.rand()
            photon.phi = 2 * np.pi * np.random.rand()
            sintheta = np.sqrt(1.0 - photon.cosz ** 2)
            photon.S1[0] = photon.S0[0] + l * sintheta * np.sin(photon.phi)
            photon.S1[1] = photon.S0[1] + l * sintheta * np.cos(photon.phi)
            photon.S1[2] = photon.S0[2] + l * photon.cosz
            # photon.S1 = np.array([14.0, 14.0, -1E-6])
            # intersection = np.array([photon.S1[0], photon.S1[1], 0.0])

            # print(photon.S0)
            # print(photon.S1)

            IS_INT0, MYNORM0, INTPOINT0 = ph.check_surface_intersections(
                        photon.S0, photon.S1,
                        x, y, Z,
                        usebox=False,
                        check_2d=False)
            # IS_INT0, MYNORM0, INTPOINT0 = intersections
            #
            IS_INT3, MYNORM3, INTPOINT3 = ph.adaptive_intersection_3(
                photon.S0, photon.S1, x, y, Z, nb1 = 10, nb2 = 6, nb3 = 6)

            IS_INT2, MYNORM2, INTPOINT2 = ph.adaptive_intersection_2(
                photon.S0, photon.S1, x, y, Z, nb1 = 10, nb2 = 10)

            if IS_INT0 or IS_INT2 or IS_INT3:
                # print(IS_INT_0)
                print('normal -> 0 = ', MYNORM0)
                print('normal -> 2 = ', MYNORM2)
                print('normal -> 3 = ', MYNORM3)
                print('inter point -> 0', INTPOINT0)
                print('inter point -> 2', INTPOINT2)
                print('inter point -> 3', INTPOINT3)
                np.testing.assert_array_almost_equal(
                    INTPOINT0, INTPOINT3, decimal=5)
                np.testing.assert_array_almost_equal(
                    INTPOINT2, INTPOINT3, decimal=5)

