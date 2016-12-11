import numpy as np
from numpy import random


def power_signal_mean(signal):
    return np.mean(a=np.multiply(signal.conj(), signal), axis=1)


def complex_randn(signal_size):
    return (random.randn(*signal_size) + 1j * random.randn(*signal_size))/np.sqrt(2)


def power_samples_in_db(array):
    array = np.real(np.multiply(array, np.conjugate(array)))
    return 10*np.log10(array)


def magnitude_to_power_db(value):
    return 20*np.log10(value)


def power_to_db(value):
    return 20*np.log10(value)


def db_to_power(value):
    return 10**(0.1*value)


def compute_throughput(snr, bandwidth=1):
    return bandwidth*np.log2(1+snr)


def norm_vectors_in_matrix(matrix, axis=0):
    norms = np.sqrt(np.sum(np.multiply(matrix, matrix.conjugate()), axis, keepdims=True))
    matrix /= norms
