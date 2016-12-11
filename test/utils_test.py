import unittest
import numpy as np
from numpy.linalg.linalg import norm

from WiSim.utils import complex_randn, norm_vectors_in_matrix


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
