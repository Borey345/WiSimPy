import numpy as np
from numpy import random


class SignalSource:

    # TYPES = {'gauss', 'qpsk', 'bit'}
    TYPE_GAUSS = 1
    TYPE_QPSK = 2
    TYPE_BIT = 3
    SQRT_2 = np.sqrt(2)

    def __init__(self, power, signal_type):
        self.signal_type = signal_type
        self.power = power
        self.magnitude = np.sqrt(power / 2)

    def get_signal(self, signal_size):
        if self.signal_type == SignalSource.TYPE_GAUSS:
            signal = (random.randn(*signal_size) + 1j * random.randn(*signal_size))

        if self.signal_type == SignalSource.TYPE_QPSK:
            i_signal = random.randint(2, size=signal_size)
            q_signal = random.randint(2, size=signal_size)
            i_signal[i_signal == 0] = -1
            q_signal[q_signal == 0] = -1
            signal = i_signal + 1j * q_signal
        if self.signal_type == SignalSource.TYPE_BIT:
            SignalSource.TYPE_BIT
            signal = random.randint(2, size=signal_size)
            return signal;

        return signal * self.magnitude


