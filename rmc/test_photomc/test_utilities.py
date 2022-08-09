
import unittest
import numpy as np
import photon_mc_numba_fractional as ph
from numba import types
from numba.typed import Dict

class TestUtilities(unittest.TestCase):

    def test_get_atm_value(self):
        """
        Test the function to extract atmospheric optics at a given z-level
        Make sure z has n+1 levels if n= number of atm layers
        Function should return the value corresponding to the lelev of the
        z value given (i.e., the value above the z level below)
        """
        dfatm = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:],
        )
        dfatm['zz'] = np.array([5, 4, 3, 2, 1, 0],   dtype=np.float64)
        dfatm['dz'] = np.array([1, 1, 1, 1, 1],      dtype=np.float64)
        dfatm['wc'] = np.array([50, 40, 30, 20, 10], dtype=np.float64)
        myz = 4.5
        res_wc = ph.get_atm_value(myz, param='wc', dfatm=dfatm)
        self.assertAlmostEqual(res_wc, 50)


    def test_meshgrid_numba(self):
        """
        Test the MyDot function
        """
        x = np.linspace(1,5, 10)
        y = np.linspace(1,3, 5)
        X1, Y1 = np.meshgrid(x, y)
        X2, Y2 = ph.meshgrid_numba(x, y)
        np.testing.assert_array_almost_equal(X1, X2, decimal = 6)
        np.testing.assert_array_almost_equal(Y1, Y2, decimal = 6)


    def test_mydot(self):
        """
        Test the MyDot function
        """
        v1 = np.array([0,0,1])
        v2 = np.array([0,2,2])
        result = ph.mydot(v1, v2)
        self.assertEqual(result, 2)


    def test_point_distance(self):
        """
        Test the MyDot function
        """
        v1 = np.array([0,0,1])
        v2 = np.array([0,0,2])
        result = ph.point_distance(v1, v2)
        self.assertEqual(result, 1)


    def test_distance_point_segment(self):
        Q = np.array([0.0, 0.0, 1.0])
        P2 = np.array([1.0, 1.0, 0.0])
        P1 = np.array([1.0, 1.0, -1.0])
        dist = ph.distance_point_segment(Q, P1, P2)
        self.assertAlmostEqual(dist, np.sqrt(3))


    def test_area_triangle(self):
        """
        Test the MyDot function
        """
        v1 = np.array([0.0,0.0,1.0])
        v2 = np.array([0.0,1.0,0.0])
        v3 = np.array([0.0,0.0,0.0])
        result = ph.area_triangle(v1, v2, v3)
        self.assertAlmostEqual(result, 0.5)


    def test_mytwonormal(self):
        """
        Test a function which given a 3d vector returns
        (1) the same vector normalized, and
        (2),(3) two normalized vectors orthogonal to the first and to each other
        """
        myv = np.array([1.0 ,2.0, 3.0])
        vn, t1n, t2n = ph.twonormal(myv)
        self.assertAlmostEqual( np.linalg.norm(vn),  1.0)
        self.assertAlmostEqual( np.linalg.norm(t2n), 1.0)
        self.assertAlmostEqual( np.linalg.norm(t1n), 1.0)
        self.assertAlmostEqual(np.dot(vn, t1n), 0.0)
        self.assertAlmostEqual(np.dot(vn, t2n), 0.0)
        self.assertAlmostEqual(np.dot(t1n, t2n), 0.0)
        self.assertAlmostEqual(np.dot(myv, vn)/np.linalg.norm(myv), 1.0)
        pass


    def test_myatan2(self):
        x = 1
        y = 1
        myatan = ph.myatan2(x, y)
        self.assertAlmostEqual(myatan, np.pi/4.0 )
        x = 1
        y = -1
        myatan = ph.myatan2(x, y)
        self.assertAlmostEqual(myatan, np.pi/4.0 + np.pi/2)
        x = -1
        y = -1
        myatan = ph.myatan2(x, y)
        self.assertAlmostEqual(myatan, np.pi/4.0 + np.pi)
        pass


    def test_polar2cart(self):
        r = 1.0
        cosz = 0.0
        phi = np.pi
        resv = ph.polar2cart(r=r, cosz=cosz, phi=phi, x0vec=(0.0, 0.0, 0.0))
        self.assertAlmostEqual(resv[2], 0.0)
        self.assertAlmostEqual(resv[1], -1.0)
        self.assertAlmostEqual(resv[0], 0.0)
        pass


    def test_cart2polar(self):
        XV = np.array([1,0,0])
        r, cosz, phi = ph.cart2polar(x=XV[0], y=XV[1], z=XV[2])
        self.assertAlmostEqual(np.linalg.norm(XV), r)
        self.assertAlmostEqual( np.cos(phi), 0.0)
        self.assertAlmostEqual( np.sin(phi), 1.0)
        self.assertAlmostEqual( cosz, 0.0)
        pass


    def test_rotate_direction(self):
        coszr = 0.0 # new direction angle relative to vector
        phir = 2.0 * np.pi * np.random.random() # new direction angle relative to vector
        VEC = np.array([0.0, 0.0, 1.0]) # vertical pointing normal
        new_cosz, new_phi = ph.rotate_direction(coszr=coszr, phir=phir, VEC=VEC)
        newvec = ph.polar2cart(r=1.0, cosz = new_cosz, phi = new_phi)
        self.assertAlmostEqual(new_cosz, 0.0)
        self.assertAlmostEqual( np.dot(newvec, VEC), 0.0)
        # self.assertAlmostEqual(new_phi, 0.0)


        coszr = 1.0 # new direction angle relative to vector
        # phir = 0.0 # new direction angle relative to vector
        phir = 2.0 * np.pi * np.random.random() # new direction angle relative to vector
        # oldvec = ph.polar2cart(r=1.0, cosz = cosz, phi = phir)
        VEC = np.array([0.0, 1.0, 0.0])
        new_cosz, new_phi = ph.rotate_direction(coszr=coszr, phir=phir, VEC=VEC)
        newvec = ph.polar2cart(r=1.0, cosz = new_cosz, phi = new_phi)
        # print(VEC)
        # print(newvec)
        # print(new_cosz, -np.pi/4 + np.arccos(coszr))
        # print(new_phi, 0.0)
        # print(np.dot(VEC, newvec)/np.linalg.norm(VEC))
        self.assertAlmostEqual( np.dot(VEC, newvec)/np.linalg.norm(VEC), 1.0)


        coszr = 1-0.01 # new direction angle relative to vector
        phir = 0.0 # new direction angle relative to vector
        # oldvec = ph.polar2cart(r=1.0, cosz = cosz, phi = phir)
        # VEC = np.array([1.0, -1.0, 2.0])
        VEC = np.array([0.0, -1.0, 0.0])
        old_r, old_cosz, old_phi = ph.cart2polar(x=VEC[0], y=VEC[1], z=VEC[2])
        new_cosz, new_phi = ph.rotate_direction(coszr=coszr, phir=phir, VEC=VEC)
        newvec = ph.polar2cart(r=1.0, cosz = new_cosz, phi = new_phi)
        # print(VEC)
        # print(newvec)
        # self.assertAlmostEqual( np.dot(VEC, newvec)/np.linalg.norm(VEC), -1.0)
        #
        # print('newvec:', newvec)
        # print('cosz:', old_cosz, new_cosz)
        # print('phi/pi:', old_phi/np.pi, new_phi/np.pi)



        # self.assertAlmostEqual(new_cosz, 0.0)
        # self.assertAlmostEqual( np.dot(newvec, VEC), 0.0)
        pass




if __name__ == '__main__':
    unittest.main()
