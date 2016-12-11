from mpl_toolkits.mplot3d.art3d import norm_angle
from scipy.linalg.basic import pinv
from scipy.linalg.decomp_svd import svd, svdvals

from WiSim.utils import complex_randn, norm_vectors_in_matrix, compute_throughput
import numpy as np
import matplotlib.pyplot as plt


def marchenko_pastur(n_dimensions: int = 8, n_realizations: int = 1000) -> None:

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


def sinr_mu_mimo(channel, precoder, noise_power=1):
    correlation = np.abs(channel.getH() * precoder.getH())**2
    n_users = channel.shape[1]
    sinr = np.zeros(n_users)
    for i_user in np.arange(n_users):
        sinr[i_user] = (correlation[i_user,i_user])/\
                       (noise_power + np.sum(correlation[i_user, np.arange(n_users) != i_user]))
    return sinr

