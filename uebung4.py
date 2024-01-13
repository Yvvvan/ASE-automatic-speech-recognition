import math

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
    num_ceps = 13
    feature = feature_extraction.compute_features(audio_file, window_size=window_size,
                                                  hop_size=hop_size,
                                                  num_ceps=num_ceps,
                                                  feature_type=feature_type)
    y_range = num_ceps if feature_type == 'MFCC' else num_ceps * 2 if feature_type == 'MFCC_D' else num_ceps * 3
    plt.imshow(feature.T, extent=[0, len(signal)/sampling_rate, y_range, 0],
               aspect='auto')
    plt.ylabel("MFCC_D_DD Index")
    plt.title("MFCC_D_DD for TEST-MAN-AH-3O33951A.wav")
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel("Time in Second")
    plt.show(aspect='auto')


if __name__ == "__main__":
    compute_features(feature_type='MFCC_D_DD')
    
    
    
