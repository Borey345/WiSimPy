import unittest
from WiSim.qam_modulaton import QamModulation
import scipy.io as sio
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import numpy as np
import matplotlib.pyplot as plt


class TestQamModulatin(unittest.TestCase):

    def testNonRegressionBpsk(self):
        self.common_qam(QamModulation.MODULATION_BPSK, 1, (0.29, 0.91, 1.91))

    def testNonRegressionQpsk(self):
        self.common_qam(QamModulation.MODULATION_QPSK, 2, (0.29, 0.91, 1.91))

    def testNonRegression16Qam(self):
        self.common_qam(QamModulation.MODULATION_16QAM, 4, (0.29, 0.91, 1.91))

    def testNonRegression64Qam(self):
        self.common_qam(QamModulation.MODULATION_64QAM, 6, (0.29, 0.91, 1.91))

    def test_qam_simple(self):
        modulator = QamModulation(QamModulation.MODULATION_64QAM)
        # bit_sequence = np.array([0, 1, 1, 1,   1, 0, 0, 0,   1, 0, 0, 1])
        bit_sequence = np.array([0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 1, 0,
                                 0, 0, 0, 0, 1, 1,
                                 0, 0, 0, 1, 0, 0,
                                 0, 0, 0, 1, 0, 1,
                                 0, 0, 0, 1, 1, 0,
                                 0, 0, 0, 1, 1, 1,
                                 0, 0, 1, 0, 0, 0,
                                 0, 0, 1, 0, 0, 1,
                                 0, 0, 1, 0, 1, 0,
                                 0, 0, 1, 0, 1, 1,
                                 0, 0, 1, 1, 0, 0,
                                 0, 0, 1, 1, 0, 1,
                                 0, 0, 1, 1, 1, 0,
                                 0, 0, 1, 1, 1, 1,
                                 0, 1, 0, 0, 0, 0,
                                 0, 1, 0, 0, 0, 1,
                                 0, 1, 0, 0, 1, 0,
                                 0, 1, 0, 0, 1, 1,
                                 0, 1, 0, 1, 0, 0,
                                 0, 1, 0, 1, 0, 1,
                                 0, 1, 0, 1, 1, 0,
                                 0, 1, 0, 1, 1, 1,
                                 0, 1, 1, 0, 0, 0,
                                 0, 1, 1, 0, 0, 1,
                                 0, 1, 1, 0, 1, 0,
                                 0, 1, 1, 0, 1, 1,
                                 0, 1, 1, 1, 0, 0,
                                 0, 1, 1, 1, 0, 1,
                                 0, 1, 1, 1, 1, 0,
                                 0, 1, 1, 1, 1, 1,
                                 1, 0, 0, 0, 0, 0,
                                 1, 0, 0, 0, 0, 1,
                                 1, 0, 0, 0, 1, 0,
                                 1, 0, 0, 0, 1, 1,
                                 1, 0, 0, 1, 0, 0,
                                 1, 0, 0, 1, 0, 1,
                                 1, 0, 0, 1, 1, 0,
                                 1, 0, 0, 1, 1, 1,
                                 1, 0, 1, 0, 0, 0,
                                 1, 0, 1, 0, 0, 1,
                                 1, 0, 1, 0, 1, 0,
                                 1, 0, 1, 0, 1, 1,
                                 1, 0, 1, 1, 0, 0,
                                 1, 0, 1, 1, 0, 1,
                                 1, 0, 1, 1, 1, 0,
                                 1, 0, 1, 1, 1, 1,
                                 1, 1, 0, 0, 0, 0,
                                 1, 1, 0, 0, 0, 1,
                                 1, 1, 0, 0, 1, 0,
                                 1, 1, 0, 0, 1, 1,
                                 1, 1, 0, 1, 0, 0,
                                 1, 1, 0, 1, 0, 1,
                                 1, 1, 0, 1, 1, 0,
                                 1, 1, 0, 1, 1, 1,
                                 1, 1, 1, 0, 0, 0,
                                 1, 1, 1, 0, 0, 1,
                                 1, 1, 1, 0, 1, 0,
                                 1, 1, 1, 0, 1, 1,
                                 1, 1, 1, 1, 0, 0,
                                 1, 1, 1, 1, 0, 1,
                                 1, 1, 1, 1, 1, 0,
                                 1, 1, 1, 1, 1, 1])
        mod = modulator.modulate(bit_sequence)
        plt.plot(mod.real, mod.imag, 'ro')
        plt.show()
        #print(modulator.demodulate(mod, 0.29))
        # print(modulator.demodulate(mod, 0.91))
        # print(modulator.demodulate(mod, 1.91))
    #
    # function
    # test16QamPerformance()
    # tic;
    # source = SignalSource(0, SignalSource.TYPE_BIT);
    # setRandomSeed(20);
    #
    # signal = source.getSignal([1, 12000000]);
    #
    # modulatorNew = QamModulation(QamModulation.MODULATION_16QAM);
    # modulatedSignalNew = modulatorNew.modulate(signal);
    # modulatorNew.demodulate(modulatedSignalNew, 0.5);
    # elapsedTime = toc;
    # assertTrue(elapsedTime < 4);
    # end
    #
    # function
    # test64QamPerformance()
    # tic;
    # source = SignalSource(0, SignalSource.TYPE_BIT);
    # setRandomSeed(20);
    #
    # signal = source.getSignal([1, 1200000]);
    #
    # modulatorNew = QamModulation(QamModulation.MODULATION_64QAM);
    # modulatedSignalNew = modulatorNew.modulate(signal);
    # modulatorNew.demodulate(modulatedSignalNew, 0.5);
    # elapsedTime = toc;
    # assertTrue(elapsedTime < 1);
    # end

    @staticmethod
    def common_qam(modulation_type, bits_per_symbol, channel_estimation, n_bits=-1):

        test_dir = 'files/qam_modulation_test'
        if n_bits == -1:
            n_bits = (2 ** 10) * bits_per_symbol

        signal = sio.loadmat(
            "%s/nonRegresssionSignalModulation_%d.mat" % (test_dir, modulation_type))['signal']
        signal = np.int8(signal)

        modulator_new = QamModulation(modulation_type)
        modulatedSignalNew = modulator_new.modulate(signal)

        modulated_signal_base = sio.loadmat('%s/nonRegressionModulation_%d' % (test_dir, modulation_type))['modulatedSignalBase']
        assert_array_almost_equal(modulated_signal_base, modulatedSignalNew)

        noise = sio.loadmat(
            "%s/nonRegresssionNoiseModulation_%d.mat" % (test_dir, modulation_type))['noise']

        modulated_signal_plus_noise_new = modulatedSignalNew + noise

        for iChannel in range(len(channel_estimation)):
            demodulated_signal_new = modulator_new.demodulate(modulatedSignalNew, channel_estimation[iChannel])
            demodulated_signal_plus_noise_new = modulator_new.demodulate(modulated_signal_plus_noise_new, channel_estimation[iChannel])

            demodulated_signal_base = sio.loadmat('%s/nonRegression_%d_%d.mat' % (test_dir, modulation_type, iChannel + 1))['demodulatedSignalBase']
            demodulated_signal_plus_noise_base = sio.loadmat('%s/nonRegressionNoise_%d_%d.mat' % (test_dir, modulation_type, iChannel + 1))['demodulatedSignalPlusNoiseBase']

            assert_array_equal(demodulated_signal_base, demodulated_signal_new)
            assert_array_equal(demodulated_signal_plus_noise_base, demodulated_signal_plus_noise_new)
