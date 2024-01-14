import numpy as np
import recognizer.tools as tools
from scipy.io import wavfile
from scipy import signal as sig
import scipy.fftpack as fftpack


def make_frames(audio_data, sampling_rate, window_size, hop_size):
    """
    :param audio_data: np.array, array of samples
    :param sampling_rate: num, samples per second
    :param window_size: float, windows size in second
    :param hop_size: float, hop size in second
    :return: array of signal sample frames (w/o Hamming)
             size = [number of frames] * [frame length]
    """
    # R (Rahmenvorschub)
    hop_size_samples = hop_size * sampling_rate
    # N (Fensterlaenge / dft_window_size)
    window_size_samples = tools.dft_window_size(window_size, sampling_rate)
    # K (Anzahl der Rahmen)
    num_frames = tools.get_num_frames(len(audio_data), window_size_samples, hop_size_samples)

    xs = []
    for i in range(num_frames):
        start = int(i * hop_size_samples)
        end = start + window_size_samples
        if end > len(audio_data):
            n = end - len(audio_data)
            xs.append(np.pad(audio_data[start:], (0, n)))  # zero padding
        else:
            xs.append(audio_data[start:end])
    # hamming window
    return np.array(xs) * np.hamming(window_size_samples)


def compute_absolute_spectrum(frames):
    """
    this function is used to calculate the absolute spectrum of a signal
    :param frames: np.array, array of signal sample frames, size = [number of frames] * [frame length]
    :return: size = [number of frames] * [frame length/2]
    """
    # magnitude = np.fft.rfft(frames)
    # return np.absolute(magnitude)

    num_frames, frame_length = frames.shape
    fft_frames = np.fft.rfft(frames, axis=1)
    # the spectrum is symmetric, get rid of the second half(redundanten Teil)
    absolute_spectrum = np.abs(fft_frames)[:, :frame_length // 2]
    return absolute_spectrum


def get_mel_filters(sampling_rate, window_size_sec, n_filters, f_min=0, f_max=8000):
    """
    the function is used to calculate the mel filters
    :param sampling_rate: in Hz
    :param window_size_sec: in second
    :param n_filters: the number of triangular filters
    :param f_min: the minimum frequency in Hz, which is covered by the filters
    :param f_max: the maximum frequency in Hz, which is covered by the filters
    :return: filters in form of np.array, size = [n_filters] * [window_size_samples/2 + 1]
    """
    mel_min = tools.hz_to_mel(f_min)
    mel_max = tools.hz_to_mel(f_max)
    # divide equally in mel scale
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    # convert back to Hz, these are the center frequencies of the filters
    freq_points = tools.mel_to_hz(mel_points)

    # window_size_samples
    N = tools.dft_window_size(window_size_sec, sampling_rate)

    # f: DFT indices, the center frequencies of the filters(freq_points) in the DFT spectrum
    # Formula from the skript, the value should be integer
    f = np.round(freq_points * N / sampling_rate).astype(int)

    weights = np.zeros((n_filters, N // 2))
    for m in range(1, n_filters + 1):
        for k in range(N // 2):
            if f[m - 1] <= k < f[m]:
                weights[m - 1, k] = 2 * (k - f[m - 1]) / ((f[m + 1] - f[m - 1]) * (f[m] - f[m - 1]))
            elif f[m] <= k <= f[m + 1]:
                weights[m - 1, k] = 2 * (f[m + 1] - k) / ((f[m + 1] - f[m - 1]) * (f[m + 1] - f[m]))
            else:
                weights[m - 1, k] = 0
    return weights


def apply_mel_filters(abs_spectrum, mel_filters):
    """
    the function is used to apply the mel filters to the spectrum,
    abs frame length = window_size_samples/2
    :param abs_spectrum: np.array, size = [number of frames] * [abs frame length]
    :param mel_filters: np.array, size = [n_filters] * [window_size_samples/2]
    :return: np.array, size = [number of frames] * [n_filters]
    """
    # np.dot: the sum of the product of the corresponding elements
    mel_spectrum = np.dot(abs_spectrum, mel_filters.T)
    return mel_spectrum


def compute_cepstrum(mel_spectrum, num_ceps):
    """
    the function is used to calculate the cepstrum of the mel spectrum
    :param mel_spectrum: np.array, size = [number of frames] * [n_filters]
    :param num_ceps: the number of cepstrum coefficients
    :return: np.array, size = [number of frames] * [num_ceps]
    """
    # replace 0 with the smallest representable value
    mel_spectrum = np.abs(mel_spectrum)
    mel_spectrum[mel_spectrum == 0] = np.finfo(float).eps
    # compute the log of the mel spectrum
    log_mel_spectrum = np.log(mel_spectrum)
    # compute the cepstrum
    cepstrum = fftpack.dct(log_mel_spectrum, axis=1, norm='ortho')[:, :num_ceps]
    return cepstrum


def get_delta(x):
    delta = np.zeros_like(x)
    delta[0] = x[1] - x[0]
    delta[-1] = x[-1] - x[-2]
    for i in range(1, delta.shape[0] - 1):
        delta[i] = (x[i + 1] - x[i - 1]) / 2
    return delta


def append_delta(x, delta):
    # output size = [number of frames] * [num_ceps * 2 (= x.num_ceps + delta.num_ceps)]
    return np.concatenate((x, delta), axis=1)


def add_context(feats, left_context=6, right_context=6):
    """
    the function is used to add context to the features
    :param feats: [f_len, f_dim], f_len: number of frames, f_dim: feature dimension
    :param left_context: the number of left context
    :param right_context: the number of right context
    :return: [f_len, f_dim , (left_context + right_context + 1)]
    """
    if left_context == 0 and right_context == 0:
        return feats

    f_len = feats.shape[0]
    f_dim = feats.shape[1]

    # Pad the sequence with copies of the first and last frames
    if left_context > 0:
        left_padding = np.tile(feats[0], (left_context, 1))
        append_feats = np.vstack([left_padding, feats])

    if right_context > 0:
        right_padding = np.tile(append_feats[-1], (right_context, 1))
        append_feats = np.vstack([append_feats, right_padding])

    context_frames = []
    for i in range(f_len):
        context_frame = append_feats[i:i+left_context+right_context+1]
        context_frames.append(context_frame)
    new_feats = np.dstack(context_frames)
    new_feats = np.swapaxes(new_feats, 0, 2)
    return new_feats


def compute_features(audio_file, window_size=25e-3, hop_size=10e-3,
                     feature_type='STFT', n_filters=24, fbank_fmin=0,
                     fbank_fmax=8000, num_ceps=13):
    sampling_rate, signal = wavfile.read(audio_file)
    signal = signal / np.max(np.abs(signal))
    signal_frames = make_frames(signal, sampling_rate, window_size=window_size, hop_size=hop_size)
    # STFT
    if feature_type == 'STFT':
        feature = compute_absolute_spectrum(signal_frames)
    elif feature_type == 'FBANK':
        feature = compute_absolute_spectrum(signal_frames)
        mel_filter = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        feature = apply_mel_filters(feature, mel_filter)
        # shall i ？
        feature[feature == 0] = np.finfo(float).eps
        feature = np.log(feature)
    elif feature_type == 'MFCC':
        feature = compute_absolute_spectrum(signal_frames)
        mel_filter = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        feature = apply_mel_filters(feature, mel_filter)
        feature = compute_cepstrum(feature, num_ceps)  # mel-log included
    elif feature_type == 'MFCC_D':
        feature = compute_absolute_spectrum(signal_frames)
        mel_filter = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        feature = apply_mel_filters(feature, mel_filter)
        feature = compute_cepstrum(feature, num_ceps)
        delta = get_delta(feature)
        feature = append_delta(feature, delta)
    elif feature_type == 'MFCC_D_DD':
        feature = compute_absolute_spectrum(signal_frames)
        mel_filter = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        feature = apply_mel_filters(feature, mel_filter)
        feature = compute_cepstrum(feature, num_ceps)
        delta = get_delta(feature)
        delta_delta = get_delta(delta)
        feature = append_delta(feature, delta)
        feature = append_delta(feature, delta_delta)
    else:
        feature = None
    return feature


def compute_features_with_context(audio_file, window_size=25e-3,
                                  hop_size=10e-3, feature_type='STFT',
                                  n_filters=24, fbank_fmin=0,
                                  fbank_fmax=8000, num_ceps=13,
                                  left_context=4, right_context=4):
    feature = compute_features(audio_file, window_size, hop_size, feature_type,
                               n_filters, fbank_fmin, fbank_fmax, num_ceps)
    feature_with_context = add_context(feature, left_context, right_context)
    return feature_with_context
