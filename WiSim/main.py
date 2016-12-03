import numpy as np
import matplotlib.pyplot as plt
from WiSim import ofdm

SNR = np.arange(-10, 41)
beta = 1
channelOn = 0
coderOn = 0
berFigure = plt.figure(1)
thFigure = plt.figure(2)

ber, per = ofdm(SNR, 1, beta, coderOn, channelOn)