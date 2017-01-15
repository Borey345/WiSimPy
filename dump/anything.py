import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from mpmath import conj
from numpy.lib.type_check import real
from scipy.linalg.decomp_svd import svdvals

from WiSim.utils import complex_randn, norm_vectors_in_matrix, compute_throughput


def marchenko_pastur(n_dimensions: int = 8, n_realizations: int =1000) -> None:

    singular_values = np.zeros((n_dimensions*n_realizations))
    for i in np.arange(0, n_realizations):
        matrix = complex_randn((n_dimensions, n_dimensions))
        singular_values[(i*n_dimensions):((i+1)*n_dimensions)] = svdvals(matrix)

    (density, bins) = np.histogram(singular_values, bins='auto', density=True)

    plt.plot(bins[:-1], density)
    plt.show()


def precoding():
    n_users = 4
    n_antennas = 8

    algorithm = 'ZF'

    n_realizations = 1000

    sinr = np.zeros((n_users, n_realizations))
    for i_realization in np.arange(n_realizations):
        H = np.matrix(complex_randn((n_antennas, n_users)))
        if algorithm == 'ZF':
            W = H.I
            norm_vectors_in_matrix(W, 1)

        sinr[:, i_realization] = sinr_mu_mimo(H, W)

    c = np.mean(sum(compute_throughput(sinr), 0))

    print('Algorithm %s gives mean %f ' % (algorithm, c))


def sinr_mu_mimo(channel: np.matrix.__class__, precoder: np.matrix.__class__, noise_power: float = 1):
    correlation = np.abs(channel.getH() * precoder.getH())**2
    n_users = channel.shape[1]
    sinr = np.zeros(n_users)
    for i_user in np.arange(n_users):
        sinr[i_user] = (correlation[i_user,i_user])/\
                       (noise_power + np.sum(correlation[i_user, np.arange(n_users) != i_user]))
    return sinr


def cholesky_decomposition(input_matrix: np.matrix.__class__):
    n_dims = input_matrix.shape[0]

    L = input_matrix.copy()

    pivots = np.empty(3)

    for i in range(n_dims):
        for j in range(i, n_dims):
            accumulator = L[i, j]

            for k in range(i-1, 0, -1):
                accumulator -= L[i, k]*conj(L[j, k])

            if i == j:
                if real(accumulator) <= 0:
                    raise ValueError('Singular or non-hermitian matrix')
                pivots[i] = sqrt(accumulator)
            else:
                L[j, i] = accumulator/pivots[i]

    return L

