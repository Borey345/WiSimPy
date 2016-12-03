import unittest

from WiSim.signal_source import SignalSource
from WiSim.utils import power_signal_mean


class TestSignalSource(unittest.TestCase):

    def test_gauss(self):

        gauss_source = SignalSource(1, SignalSource.TYPE_GAUSS)

        signal = gauss_source.get_signal((1, 10000))

        power = power_signal_mean(signal)
        image_part = power[0].imag
        real_part = power[0].real
        self.assertTrue((real_part > 0.9) and (real_part < 1.1))
        self.assertTrue((image_part > -0.1) and (image_part < 0.1))

        gauss_source = SignalSource(3, SignalSource.TYPE_GAUSS)

        power = power_signal_mean(gauss_source.get_signal((1, 10000)))
        image_part = power[0].imag
        real_part = power[0].real
        self.assertTrue((real_part > 2.9) and (real_part < 3.1))
        self.assertTrue((image_part > -0.1) and (image_part < 0.1))

    def test_qpsk(self):
        qpsk_source = SignalSource(1, SignalSource.TYPE_QPSK)

        signal = qpsk_source.get_signal((1, 10))

        power = power_signal_mean(signal)
        image_part = power[0].imag
        real_part = power[0].real
        self.assertTrue((real_part > 0.9) and (real_part < 1.1))
        self.assertTrue((image_part > -0.1) and (image_part < 0.1))

        qpsk_source = SignalSource(3, SignalSource.TYPE_QPSK)

        power = power_signal_mean(qpsk_source.get_signal((1, 10000)))
        image_part = power[0].imag
        real_part = power[0].real
        self.assertTrue((real_part > 2.9) and (real_part < 3.1))
        self.assertTrue((image_part > -0.1) and (image_part < 0.1))

        qpsk_source = SignalSource(2, SignalSource.TYPE_QPSK)
        signal = qpsk_source.get_signal(1)
        real_part = signal[0].real
        image_part = signal[0].imag

        self.assertTrue(real_part == 1 or real_part == -1)
        self.assertTrue(image_part == 1 or image_part == -1)

    def test_bit(self):
        bit_source = SignalSource(0, SignalSource.TYPE_BIT)

        signal = bit_source.get_signal(10)

        for s in signal:
            self.assertTrue(s == 1 or s == 0)








