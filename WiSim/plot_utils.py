import numpy as np
from numpy.fft.fftpack import fft
from numpy.fft.helper import fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy.signal.filter_design import freqz

from WiSim.utils import power_samples_in_db


def plot_dft_real_image(signal: np.array.__class__, frequency_discretization=1, is_in_db=False):

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


def plot_dft_abs_phase(signal: np.array.__class__, frequency_discretization=1, is_in_db=False):
    spectrum = fftshift(fft(signal))

    fft_frequencies = fftshift(fftfreq(signal.shape[0], 1 / frequency_discretization))

    abs_values = np.abs(spectrum)
    if is_in_db:
        abs_values = power_samples_in_db(abs_values)

    plt.figure()

    plt.subplot(211)
    plt.plot(fft_frequencies, abs_values, color='b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency')
    plt.grid()

    plt.subplot(212)
    plt.plot(fft_frequencies, np.angle(spectrum), color='g')
    plt.ylabel('Phase [rad]', color='g')
    plt.xlabel('Frequency')
    plt.grid()

    plt.show()


def plot_filter_responce(filter_coefficients):
    w, h = freqz(filter_coefficients)

    fig = plt.figure()
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)

    plt.plot(w, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()
