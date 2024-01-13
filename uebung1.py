import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from recognizer import feature_extraction
import numpy as np


def compute_features():
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    sampling_rate, signal = wavfile.read(audio_file)
    # window_size, hop_size = 25e-3, 10e-3
    window_size, hop_size = 0.4, 0.25
    signal_frames = feature_extraction.make_frames(signal, sampling_rate, window_size=window_size, hop_size=hop_size)

    # normalized between 1 and -1
    # min_val = min(np.min(signal_frames), -np.max(signal_frames))
    # max_val = -min_val
    # signal_frames = 2 * (signal_frames - min_val) / (max_val - min_val) - 1

    # only show the first 4 frames
    signal_frames = signal_frames[:4]
    num_signal_frames = len(signal_frames)

    # show frames
    f, axs = plt.subplots(num_signal_frames, 1, figsize=(15, 2 * num_signal_frames))
    f.suptitle("First 4 successive frames (normalized between -1 and 1) with Hamming\n "
               "with window size {}ms and hop size {}ms".format(window_size * 1000, hop_size * 1000))
    x_axis = np.arange(0, len(signal_frames[0]) / sampling_rate, 1 / sampling_rate)
    for i in range(num_signal_frames):
        axs[i].plot(x_axis, signal_frames[i])
        axs[i].grid()
        axs[i].set_xlim((0, len(signal_frames[0]) / sampling_rate))
    plt.xlabel("Time in Seconds")
    plt.show()


if __name__ == "__main__":
    compute_features()
