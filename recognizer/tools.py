# import...

def sec_to_samples(x, sampling_rate):
    return x * sampling_rate


def next_pow2(x):
    p = 0
    while True:
        if x > 2 ** p:
            p += 1
        else:
            return p


def next_pow2_samples(x, sampling_rate):
    # TODO implement this method
    pass


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    # TODO implement this method
    pass
