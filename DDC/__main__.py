import numpy as np
from numpy.fft.helper import fftfreq, fftshift
from scipy.fftpack import ifft, ifftshift
from scipy.signal.signaltools import convolve

from WiSim.plot_utils import plot_spectrum_real_image
from WiSim.signal_source import SignalSource
from WiSim.utils import complex_randn, db_to_power

frequency_discretization = 600 * 10 ** 6

period_discretization = 1 / frequency_discretization

window_length = 4*1024

fft_frequencies = fftshift(fftfreq(window_length, period_discretization))

signal_borders = (70 * 10 ** 6, 230 * 10 ** 6)
signal_positions = (fft_frequencies > signal_borders[0]) & (fft_frequencies < signal_borders[1])
n_signal_positions = sum(signal_positions)

random_component = complex_randn((n_signal_positions,))
coefficients = np.arange(1, random_component.shape[0] + 1)
random_component *= coefficients
signal = np.zeros((window_length,), dtype=np.complex128)
signal[signal_positions] = random_component


signal = ifft(ifftshift(signal))

snr = -20
snr = db_to_power(snr)
noise = SignalSource(snr, SignalSource.TYPE_GAUSS).get_signal(signal.shape)
signal += noise

signal = np.complex128(np.real(signal))

first_signal_position = signal_positions.ravel().nonzero()[0][0] - int(window_length / 2)
n_samples_shift = first_signal_position + int(n_signal_positions / 2)
exponential = np.exp(-2j * np.pi * (n_samples_shift / window_length) * np.arange(0, window_length))
signal *= exponential  # move spectrum to zero frequency

signal = convolve(signal, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1]), mode='same')

plot_spectrum_real_image(signal, frequency_discretization, is_in_db=True)
# input("Press Enter to finish...")

