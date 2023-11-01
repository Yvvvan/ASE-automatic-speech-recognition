# import...
# z.B.: import recognizer.tools as tools
import numpy as np
from scipy.io import wavfile


def make_frames(audio_data, sampling_rate, window_size, hop_size):
    # audio_data np.array
    xs = []
    start = 0
    while start < len(audio_data):
        end = int (start + window_size * sampling_rate)
        if end > len(audio_data):
            n = end - len(audio_data)
            xs.append(np.pad(audio_data[start:],(0,n)) * np.hamming(window_size*sampling_rate))
        else:
            xs.append(audio_data[start:end]* np.hamming(window_size*sampling_rate))
        start = int (start + hop_size * sampling_rate)

    return xs
