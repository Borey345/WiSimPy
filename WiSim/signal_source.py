import numpy as np
from numpy import random

from WiSim.utils import complex_randn


class SignalSource:

    TYPE_GAUSS = 1
    TYPE_QPSK = 2
    TYPE_BIT = 3
    SQRT_2 = np.sqrt(2)

    def __init__(self, power, signal_type):
        self.signal_type = signal_type
        self.power = power
        self.magnitude = np.sqrt(power)

    def get_signal(self, signal_size):
        if self.signal_type == SignalSource.TYPE_GAUSS:
            return self.magnitude*complex_randn(signal_size)
        if self.signal_type == SignalSource.TYPE_QPSK:
            i_signal = random.randint(2, size=signal_size)
            q_signal = random.randint(2, size=signal_size)
            i_signal[i_signal == 0] = -1
            q_signal[q_signal == 0] = -1
            signal = (i_signal + 1j * q_signal)/SignalSource.SQRT_2
        if self.signal_type == SignalSource.TYPE_BIT:
            SignalSource.TYPE_BIT
            return random.randint(2, size=signal_size)

        return signal * self.magnitude


