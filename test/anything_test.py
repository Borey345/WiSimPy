import unittest

import dump.anything as any
from WiSim.utils import complex_randn


class TestAnything(unittest.TestCase):
    def test_cholesky(self):
        H = complex_randn((3,3))
        R = H.dot(H.transpose().conj())

