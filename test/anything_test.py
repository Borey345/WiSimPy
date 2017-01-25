import unittest

from numpy import random

import dump.anything as any
from WiSim.utils import complex_randn
import numpy as np
import numpy.testing as np_test


class TestAnything(unittest.TestCase):
    def test_cholesky_result_is_lower_triangular(self):
        n_dimensions = 3
        H = complex_randn((n_dimensions, n_dimensions))
        R = H.dot(H.transpose().conj()) + np.identity(n_dimensions)

        R_chol = any.cholesky_decomposition(R)

        for i in np.arange(0, n_dimensions):
            for j in np.arange(i+1, n_dimensions):
                self.assertEqual(R_chol[i, j], 0)

    def test_cholesky_multiplication_gives_original_matrix_real(self):
        n_dimensions = 3
        H = random.randn(n_dimensions, n_dimensions)
        R = H.dot(H.transpose().conj()) + np.identity(n_dimensions)

        R_chol = any.cholesky_decomposition(R)

        np_test.assert_array_almost_equal(R_chol.dot(R_chol.transpose().conj()), R)

    def test_cholesky_multiplication_gives_original_matrix_complex(self):
        n_dimensions = 3
        H = complex_randn((n_dimensions, n_dimensions))
        R = H.dot(H.transpose().conj())  + np.identity(n_dimensions)

        R_chol = any.cholesky_decomposition(R)

        np_test.assert_array_almost_equal(R_chol.dot(R_chol.transpose().conj()), R)

    def test_result_0_0_element_is_square_root_of_0_0_original_element(self):
        n_dimensions = 3
        H = complex_randn((n_dimensions, n_dimensions))
        R = H.dot(H.transpose().conj()) + np.identity(n_dimensions)

        R_chol = any.cholesky_decomposition(R)

        self.assertAlmostEqual(R_chol[0, 0]*R_chol[0, 0], R[0, 0])


