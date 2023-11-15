import numpy as np
import recognizer.tools as tools


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

def compute_absolute_spectrum(frames):
    return

if __name__ == "__main__":
    from scipy.io import wavfile
    audio_file = '../data/TEST-MAN-AH-3O33951A.wav'
    sampling_rate, signal = wavfile.read(audio_file)
    signal_frames = make_frames(signal, sampling_rate, window_size=0.4, hop_size=0.25)
    debug = "here"