import math

import numpy as np


def sec_to_samples(x, sampling_rate):
    return int(x * sampling_rate)


def next_pow2(x):
    return math.ceil(math.log(x, 2))


def dft_window_size(x, sampling_rate):
    x_len = sec_to_samples(x, sampling_rate)
    p = next_pow2(x_len)
    return 2 ** p


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    """
    this function is used to calculate the number of total frames in a signal
    :param signal_length_samples: the length of the signal in samples
    :param window_size_samples:   the length of the window in samples
    :param hop_size_samples:      the length of the hop size (not overlapped area) in samples
    :return: how many frames in total
    """
    return math.ceil((signal_length_samples - (window_size_samples - hop_size_samples)) / hop_size_samples)


def hz_to_mel(x):
    """
    this function is used to convert frequency from Hz to Mel
    :param x: the frequency in Hz or an array of frequencies in Hz
    :return: mel value
    """
    return 2595*np.log(1+x/700)


def mel_to_hz(x):
    return 700*(np.exp(x/2595)-1)
