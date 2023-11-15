import math

import matplotlib.pyplot as plt
from scipy.io import wavfile
from recognizer import feature_extraction
import numpy as np
import pylab as pl
import numpy as np
from PIL import Image


def compute_features():
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    sampling_rate, signal = wavfile.read(audio_file)
    signal_frames = feature_extraction.make_frames(signal, sampling_rate, window_size=0.4, hop_size=0.25)

    feature = feature_extraction.compute_features(audio_file)
    feature = 20*np.log10(feature)
    plt.imshow(feature.T, extent=[0, len(signal)/sampling_rate, 8000, 0],
               aspect='auto')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.ylabel("Freqency in Hz")
    plt.xlabel("Time in Second")
    plt.show(aspect='auto')


if __name__ == "__main__":
    ################
    # SPEKTRALANALYSE
    ################
    compute_features()
    
    ################
    # DREIECKSFILTER
    ################
    
    
    
    
    ##############
    # MEL-SPEKTRUM
    ##############
    
    
    
