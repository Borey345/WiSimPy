import numpy as np
from numpy import random


def power_signal_mean(signal):
    return np.mean(a=np.multiply(signal.conj(), signal), axis=1)


def complex_randn(signal_size):
    return (random.randn(*signal_size) + 1j * random.randn(*signal_size)) / np.sqrt(2)


def power_samples_in_db(array):
    array = np.real(np.multiply(array, np.conjugate(array)))
    return 10 * np.log10(array)


def magnitude_to_power_db(value):
    return 20 * np.log10(value)


def power_to_db(value):
    return 20 * np.log10(value)


def db_to_power(value):
    return 10 ** (0.1 * value)


def compute_throughput(snr, bandwidth=1):
    return bandwidth * np.log2(1 + snr)


def norm_vectors_in_matrix(matrix, axis=0):
    norms = np.sqrt(np.sum(np.multiply(matrix, matrix.conjugate()), axis, keepdims=True))
    matrix /= norms


def frequencies_matrix(freqencies_row: np.array.__class__, time_instants_column: np.array.__class__):
    return np.exp(2j * np.pi * freqencies_row * time_instants_column)


def steering_vector(angles_row: np.array.__class__, antenna_distance: float = 0.5, n_antennas=0,
                    antenna_positions: np.array.__class__ = []):
    if not antenna_positions:
        if n_antennas == 0:
            raise ValueError('n_antennas or antenna_positions must be specified')

        antenna_positions = np.arange(0, antenna_distance * n_antennas, antenna_distance)

        return frequencies_matrix(np.sin(angles_row.reshape(1, -1)), antenna_positions.reshape(-1, 1))
