# import...
# z.B.: import recognizer.feature_extraction as fe
import matplotlib.pyplot as plt
from scipy.io import wavfile
from recognizer import feature_extraction
import matplotlib

def compute_features():
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    sampling_rate, signal = wavfile.read(audio_file)
    xs = feature_extraction.make_frames(signal, sampling_rate, 0.25, 0.1)
    print(len(xs))
    '''plt.plot(range(len(xs[0])), xs[0])
    plt.show()'''



if __name__ == "__main__":
    compute_features()

