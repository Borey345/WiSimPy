import unittest
import numpy as np
from numpy.linalg.linalg import norm
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

from WiSim.utils import complex_randn, norm_vectors_in_matrix, steering_vector


class TestUtils(unittest.TestCase):

    def test_norm_vectors_in_matrix_default(self):
        n_vectors = 4
        mx = complex_randn((8, n_vectors))
        norm_vectors_in_matrix(mx)
        for i_vector in np.arange(n_vectors):
            self.assertAlmostEqual(1, norm(mx[:, i_vector]))

    def test_norm_vectors_in_matrix_row(self):
        n_vectors = 4
        n_rows = 8
        mx = complex_randn((n_rows, n_vectors))
        norm_vectors_in_matrix(mx, 1)
        for i_row in np.arange(n_rows):
            self.assertAlmostEqual(1, norm(mx[i_row, :]))

    def test_steering_vector(self):
        A = steering_vector(
            np.array([0, np.pi/6, np.pi/2]), n_antennas=4)
        expected = np.array([[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1]])
        assert_array_almost_equal(expected.transpose(), A)
