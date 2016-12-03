import numpy as np


def power_signal_mean(signal):
    return np.mean(a=np.multiply(signal.conj(), signal), axis=1)
