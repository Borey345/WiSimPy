from scipy import signal

from WiSim.plot_utils import plot_filter_responce
import numpy as np


filter_coefs = signal.firwin(128, 80*10**6, nyq=300*10**6, window=('kaiser', 6))
plot_filter_responce(filter_coefs)

file_name = 'filter_coefficients.npy'
np.save(file_name, filter_coefs)

filter_coefs_loaded = np.load(file_name)

plot_filter_responce(filter_coefs_loaded)



# plot_dft_abs_phase(coefficients, is_in_db=True)
