import numpy as np
from typing import NewType

class PhaseLockedLoop:

    __frequency_vco

    def __init__(self, frequency_nominal_vco: float, frequency_discretization: float = 0):
        self.frequency_vco = frequency_nominal_vco
        return

    def lock(self, signal: np.array.__class__):
        out_frequency = np.zeros((3,3))
        return signal

    def __detect_phase(self, signal):
        for i_sample in np.arange(0, signal.shape[1]):









