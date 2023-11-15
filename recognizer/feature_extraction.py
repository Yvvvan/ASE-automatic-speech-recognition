import numpy as np
import recognizer.tools as tools
from scipy.io import wavfile
from scipy import signal as sig


def make_frames(audio_data, sampling_rate, window_size, hop_size):
    """
    :param audio_data: np.array, array of samples
    :param sampling_rate: int, samples per second
    :param window_size: float, windows size in second
    :param hop_size: float, hop size in second
    :return: array of signal sample frames (w/o Hamming)
             size = number of frames * frame length
    """
    # R (Rahmenvorschub)
    hop_size_samples = hop_size * sampling_rate
    # N (Fensterlaenge)
    dft_window_size = tools.dft_window_size(window_size, sampling_rate)
    window_size_samples = dft_window_size
    # K (Anzahl der Rahmen)
    num_frames = tools.get_num_frames(len(audio_data), window_size_samples, hop_size_samples)

    xs = []
    for i in range(num_frames):
        start = int(i * hop_size_samples)
        end = start + window_size_samples
        if end > len(audio_data):
            n = end - len(audio_data)
            xs.append(np.pad(audio_data[start:], (0, n)))
        else:
            xs.append(audio_data[start:end])

    return np.array(xs) * np.hamming(window_size_samples)


def normalization(signal_frames):
    # normalized between 1 and -1
    min_val = min(np.min(signal_frames), -np.max(signal_frames))
    max_val = -min_val
    signal_frames = 2 * (signal_frames - min_val) / (max_val - min_val) - 1
    return signal_frames


def compute_absolute_spectrum(frames):
    magnitude = np.fft.rfft(frames)
    return np.absolute(magnitude)


def compute_features(audio_file, window_size=25e-3, hop_size=10e-3,
                     feature_type='STFT', n_filters=24, fbank_fmin=0,
                     fbank_fmax=8000, num_ceps=13):
    sampling_rate, signal = wavfile.read(audio_file)
    signal = normalization(signal)
    signal_frames = make_frames(signal, sampling_rate, window_size=window_size, hop_size=hop_size)
    # STFT
    if feature_type == 'STFT':
        feature = compute_absolute_spectrum(signal_frames)
    else:
        feature = None
    return feature


def get_mel_filters(sampling_rate, window_size_sec,
                    n_filters, f_min=0, f_max=8000):
    mel_min = tools.hz_to_mel(f_min)
    mel_max = tools.hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)