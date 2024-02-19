import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from recognizer import feature_extraction
import numpy as np
import pylab as pl
import numpy as np
from PIL import Image


def compute_features(feature_type='STFT'):
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    sampling_rate, signal = wavfile.read(audio_file)
    window_size = 25e-3
    hop_size = 10e-3
    n_filters = 24
    feature = feature_extraction.compute_features(audio_file, window_size=window_size,
                                                  hop_size=hop_size, n_filters=n_filters, feature_type=feature_type)

    if feature_type == 'STFT':
        feature = 20 * np.log10(feature)
        # feature.T makes the frequency axis vertical and the time axis horizontal
        # extent=[0, len(signal)/sampling_rate, 8000, 0] sets the axis range
        # 0, len(signal)/sampling_rate is the time axis range
        # 8000, 0 is the frequency axis range
        plt.imshow(feature.T, extent=[0, len(signal)/sampling_rate, 8000, 0],
                   aspect='auto')
        plt.ylabel("Frequency in Hz")
        plt.title("Spectrogram for TEST-MAN-AH-3O33951A.wav")
    elif feature_type == 'FBANK':
        plt.imshow(feature.T, extent=[0, len(signal)/sampling_rate, n_filters, 0],
                   aspect='auto')
        plt.ylabel("Mel filter index")
        plt.title("Mel-Spektrum for TEST-MAN-AH-3O33951A.wav")

    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel("Time in Second")
    plt.show()


def show_mel_filters():
    sampling_rate = 16e3
    window_size = 25e-3
    n_filters = 24
    mel_filter = feature_extraction.get_mel_filters(sampling_rate, window_size, n_filters)
    for i in range(n_filters):
        plt.plot(mel_filter[i])
    plt.title("Mel Filter Bank")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency in Hz")
    plt.show()


if __name__ == "__main__":
    ################
    # SPEKTRALANALYSE
    ################
    # uebung 2
    # compute_features()
    
    ################
    # DREIECKSFILTER
    ################
    show_mel_filters()
    
    
    ##############
    # MEL-SPEKTRUM
    ##############
    compute_features(feature_type='FBANK')
    
    
    
