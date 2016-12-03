import numpy as np


class QamModulation:
    MODULATION_BPSK = 1
    MODULATION_QPSK = 2
    MODULATION_16QAM = 3
    MODULATION_64QAM = 4

    NORMALIZE_COEFFICIENT_BPSK = 1 / np.sqrt(2)
    NORMALIZE_COEFFICIENT_16QAM = np.sqrt(10)

    _modulation = MODULATION_BPSK
    _isSoft = 0

    def __init__(self, modulation):
        self._modulation = modulation

    def modulate(self, bit_stream):
        if self._modulation == QamModulation.MODULATION_BPSK:
            signal = (2 * bit_stream - 1)
        elif self._modulation == QamModulation.MODULATION_QPSK:
            bit_stream = bit_stream.reshape(-1, 2).transpose()
            b = bit_stream[0,:]
            c = bit_stream[1,:]
            signal = (2 * b - 1) + 1j * (2 * c - 1)
            signal *= QamModulation.NORMALIZE_COEFFICIENT_BPSK
        elif self._modulation == QamModulation.MODULATION_16QAM:
            bit_stream = bit_stream.reshape(-1, 4).transpose()
            b = bit_stream[0,:]
            c = bit_stream[1,:]
            d = bit_stream[2,:]
            e = bit_stream[3,:]
            z = 2 * d + e
            y = 2 * b + c
            signal = ((-4 / 3) * (y**3) + 5 * (y**2) - (5 / 3) * y - 3) + \
                     1j * ((-4 / 3) * (z**3) + 5 * (z**2) - (5 / 3) * z - 3)
            signal /= np.sqrt(10)
        elif self._modulation == QamModulation.MODULATION_64QAM:
            bit_stream = bit_stream.reshape(-1, 6).transpose()
            b = bit_stream[0,:]
            c = bit_stream[1,:]
            d = bit_stream[2,:]
            e = bit_stream[3,:]
            g = bit_stream[4,:]
            h = bit_stream[5,:]
            x = (4 * b + 2 * c + d).astype(float)
            y = (4 * e + 2 * g + h).astype(float)
            a1 = 0.0508 * (x**7)
            # a1_ = 0.0508 * np.power(x, 7)
            a2 = 1.2667 * (x**6)
            a3 = 12.4556 * (x**5)
            a4 = 61.0833 * (x**4)
            a5 = 155.1556 * (x**3)
            # a5_ = 155.1556 * np.power(x, 3)
            a6 = 189.6500 * (x**2)
            a7 = 82.3381 * x
            f_real = -a1 + a2 - a3 + a4 - a5 + a6 - a7 - 7
            f_imag = (-0.0508 * (y**7) + 1.2667 * (y**6) - 12.4556 * (y**5) + 61.0833 * (y**4) - 155.1556 * (y**3) +
                      189.6500 * (y**2) - 82.3381 * y - 7)
            sz = f_real.shape[0]
            for jj in range(sz):
                if 0.8504 <= f_real[jj] <= 0.8506:
                    f_real[jj] = 3
                if 0.8504 <= f_imag[jj] <= 0.8506:
                    f_imag[jj] = 3

            signal = np.ceil(f_real) + 1j * np.ceil(f_imag)
            signal = signal / np.sqrt(42)
            print(signal)
        else:
            raise ValueError('Unknown modulation')
        return signal.reshape(1, -1)

    def demodulate(self, signal, channel_coefficients):
        signal = signal[:]
        if self._modulation == QamModulation.MODULATION_BPSK:
            signal = signal.real
            bit_stream = QamModulation.limits(signal)
        elif self._modulation == QamModulation.MODULATION_QPSK:
            in_phase = (np.sqrt(2) * signal).real
            quadrature = (np.sqrt(2) * signal).imag
            in_phase = in_phase.reshape(-1, 1)
            quadrature = quadrature.reshape(-1, 1)
            bit_stream = np.hstack((in_phase, quadrature))
            bit_stream = bit_stream.reshape(1, -1)
            bit_stream = QamModulation.limits(bit_stream)

        elif self._modulation == QamModulation.MODULATION_16QAM:
            in_phase = np.sqrt(10) * signal.real
            quadrature = np.sqrt(10) * signal.imag
            in_phase = in_phase.reshape(-1, 1)
            quadrature = quadrature.reshape(-1, 1)

            in_phase_quadrature = np.hstack((in_phase,  quadrature))
            bit1_array = in_phase_quadrature.copy()

            more_than2 = bit1_array > 2
            bit1_array[more_than2] = 2 * (bit1_array[more_than2] - 1)
            less_than2 = bit1_array < -2
            bit1_array[less_than2] = 2 * (bit1_array[less_than2] + 1)

            bit2Array = in_phase_quadrature
            bit2Array = -abs(bit2Array) + 2
            bit_stream = np.vstack((bit1_array[:, 0], bit2Array[:, 0], bit1_array[:, 1], bit2Array[:, 1]))
            bit_stream = bit_stream.transpose().reshape(1, -1)
            bit_stream = np.multiply(bit_stream, channel_coefficients) / 4
            bit_stream = QamModulation.limits(bit_stream)

        elif self._modulation == QamModulation.MODULATION_64QAM:
            in_phase = (np.sqrt(42) * signal).real
            quadrature = (np.sqrt(42) * signal).imag
            in_phase = in_phase.reshape(-1, 1)
            quadrature = quadrature.reshape(-1, 1)

            inPhaseBit = QamModulation.qam64_soft_demod(in_phase)
            quadratureBit = QamModulation.qam64_soft_demod(quadrature)

            bit_stream = np.hstack((inPhaseBit, quadratureBit))
            bit_stream = bit_stream.reshape(1, -1)
            bit_stream = np.multiply(bit_stream, channel_coefficients) / 4
            bit_stream = QamModulation.limits(bit_stream)
        else:
            raise ValueError('Unknown modulation')

        return bit_stream

    @staticmethod
    def limits(in_signal):

        out = np.floor((7 / 2) * (in_signal + 1) + 0.5)
        out[out > 7] = 7
        out[out < 0] = 0
        return np.int8(out)

    @staticmethod
    def qam64_soft_demod(in_signal):
        out = np.zeros((3, in_signal.shape[0]))

        current_diapason_values_indices = in_signal > 6
        current_diapason_values = in_signal[current_diapason_values_indices]
        out[0, current_diapason_values_indices] = 4 * (current_diapason_values - 3)
        out[1, current_diapason_values_indices] = 2 * (-current_diapason_values + 5)
        out[2, current_diapason_values_indices] = -current_diapason_values + 6

        current_diapason_values_indices = in_signal < -6
        current_diapason_values = in_signal[current_diapason_values_indices]
        out[0, current_diapason_values_indices] = 4 * (current_diapason_values + 3)
        out[1, current_diapason_values_indices] = 2 * (current_diapason_values + 5)
        out[2, current_diapason_values_indices] = current_diapason_values + 6

        current_diapason_values_indices = (in_signal > 4) & (in_signal <= 6)
        current_diapason_values = in_signal[current_diapason_values_indices]
        out[0, current_diapason_values_indices] = 3 * (current_diapason_values - 2)
        out[1, current_diapason_values_indices] = 4 - current_diapason_values
        out[2, current_diapason_values_indices] = -current_diapason_values + 6

        current_diapason_values_indices = (in_signal >= -6) & (in_signal < -4)
        current_diapason_values = in_signal[current_diapason_values_indices]
        out[0, current_diapason_values_indices] = 3 * (current_diapason_values + 2)
        out[1, current_diapason_values_indices] = 4 + current_diapason_values
        out[2, current_diapason_values_indices] = current_diapason_values + 6

        current_diapason_values_indices = (in_signal > 2) & (in_signal <= 4)
        current_diapason_values = in_signal[current_diapason_values_indices]
        out[0, current_diapason_values_indices] = 2 * (current_diapason_values - 1)
        out[1, current_diapason_values_indices] = 4 - current_diapason_values
        out[2, current_diapason_values_indices] = current_diapason_values - 2

        current_diapason_values_indices = (in_signal >= -4) & (in_signal < -2)
        current_diapason_values = in_signal[current_diapason_values_indices]
        out[0, current_diapason_values_indices] = 2 * (current_diapason_values + 1)
        out[1, current_diapason_values_indices] = 4 + current_diapason_values
        out[2, current_diapason_values_indices] = -current_diapason_values - 2

        current_diapason_values_indices = (in_signal >= -2) & (in_signal <= 2)
        current_diapason_values = in_signal[current_diapason_values_indices]
        out[0, current_diapason_values_indices] = current_diapason_values
        out[1, current_diapason_values_indices] = 2 * (-abs(current_diapason_values) + 3)
        out[2, current_diapason_values_indices] = abs(current_diapason_values) - 2
        return out

    @staticmethod
    def n_modulated_bits_to_modulation_type(n_modulated_bits):
        if n_modulated_bits == 1:
            modulation_type = QamModulation.MODULATION_BPSK
        elif n_modulated_bits == 2:
            modulation_type = QamModulation.MODULATION_QPSK
        elif n_modulated_bits == 4:
            modulation_type = QamModulation.MODULATION_16QAM
        elif n_modulated_bits == 6:
            modulation_type = QamModulation.MODULATION_64QAM
        else:
            raise ValueError('Unknown modulation')
        return modulation_type
