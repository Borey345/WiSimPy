from WiSim.signal_source import SignalSource
from WiSim.progress import print_progress
import numpy as np
from WiSim.qam_modulaton import QamModulation


def  ofdm(snr_values, modulation_type=2, beta=0.3, coder_on=1, channel_on=1):

    n_subcarriers = 48
    n_realiz = 1
    if channel_on:
        nbits = 2 ^ 4 * n_subcarriers
        n_channel = 200
        n_noise = 20
        if modulation_type == 6:
            nbits = 2 ^ 4 * n_subcarriers
            n_channel = 100
    else :
        nbits = 2 ^ 10 * n_subcarriers
        n_channel = 1
        n_noise = 1

    packet_length = 48 * 2
    counter = 0
    # SNR = zeros(22, 1)

    bit_source = SignalSource(1, SignalSource.TYPE_BIT)
    modulation = QamModulation(QamModulation.n_modulated_bits_to_modulation_type(modulation_type))
    ofdmModulation = OfdmModulation(0)

    n_points = snr_values.shape[1]
    print_progress(0, n_points)

    per = np.zeros((n_realiz, n_noise, n_channel))
    ber_coded = np.zeros((n_realiz, n_noise, n_channel))

    mean_ber_coded = np.zeros(snr_values.shape[1])
    mean_per = np.zeros(snr_values.shape[1])
    for snr in snr_values:
        counter+=1
        noise_source = SignalSource(10 ** (-snr / 10), SignalSource.TYPE_GAUSS)

        for realisation in range(n_realiz):

            bits = bit_source.getSignal([1, modulation_type * nbits])
            if coder_on:
                coded_bits = Coder_Wi_Fi(bits)
            else:
                coded_bits = bits

            # coded_bits = interleaver(coded_bits, modulation_type, n_subcarriers);
            modulated_signal = modulation.modulate(coded_bits)
            mapped_signal_original = mapper(modulated_signal, n_subcarriers)

            time_domain_signal = ofdmModulation.modulate(mapped_signal_original)

            for noise_realiz in range(n_noise):

                noise = noise_source.getSignal(time_domain_signal.shape)

                for channel_realiz in range(n_channel):

                    mapped_signal = mapped_signal_original.copy()

                    # if channel_on
                    #     H = channelCoefficients(beta);
                    # else
                    H = np.ones(1, 64)
                    # end
                    #
                    #
                    # if channel_on
                    #     for i=1:64
                    # mapped_signal(i,: ) = H(1, i) * mapped_signal(i,: );
                    # end
                    # end

                    signal_with_noise = time_domain_signal + noise

                    # % if channel_on
                    #     % for i=1:64
                    # % signal_with_noise(i,: ) = signal_with_noise(i,: ) / H(1, i);
                    # % end
                    # % end

                    signal_with_noise = ofdmModulation.demodulate(signal_with_noise)

                    demappedSignal = demapper(signal_with_noise)

                    channelPowers = channelPowerForDemappedSignal(H.conj().transpose(),
                                                                  modulation_type, demappedSignal.shape[1])

                    demodulatedSignal = modulation.demodulate(demappedSignal, channelPowers)
                    # % demodulatedSignal = demappedSignal;
                    # % demodulatedSignal = deinterleaver(demodulatedSignal, modulation_type, n_subcarriers);
                    if coder_on:
                        out_bits = decoder(demodulatedSignal);
                    else:
                        out_bits = demodulatedSignal > 3

                    errors = out_bits != bits
                    per[realisation, noise_realiz, channel_realiz] = np.mean(np.sum(errors.reshape(packet_length, -1), 0) != 0)
                    ber_coded[realisation, noise_realiz, channel_realiz] = np.mean(errors)

        mean_ber_coded[counter] = np.mean(ber_coded)
        mean_per[counter] = np.mean(per)
        print_progress(counter, n_points)

    return mean_ber_coded, mean_per


