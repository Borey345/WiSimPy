import numpy as np
from numpy.fft.fftpack import fft
from numpy.fft.helper import fftshift, fftfreq
import matplotlib.pyplot as plt

from WiSim.utils import power_samples_in_db


def plot_spectrum_real_image(signal, frequency_discretization=1, is_in_db=False):

    spectrum = fftshift(fft(signal))

    if is_in_db:
        spectrum = power_samples_in_db(np.real(spectrum)) + 1j*power_samples_in_db(np.imag(spectrum))

    fft_frequencies = fftshift(fftfreq(signal.shape[0], 1/frequency_discretization))
    plt.subplot(211)
    plt.plot(fft_frequencies, np.real(spectrum))
    plt.subplot(212)
    plt.plot(fft_frequencies, np.imag(spectrum))
    # plt.ion()
    plt.show()

