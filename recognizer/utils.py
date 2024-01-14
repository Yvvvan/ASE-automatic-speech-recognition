import os
import json
import numpy
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import random
from recognizer import tools
from recognizer import feature_extraction as fe
from recognizer import hmm as HMM

SEED = 666
random.seed(SEED)


def get_data(datadir):
    """
    get_data() load the meta data in dictionary. 
    Input:
        datadir: <string> the folder saves the meta data
    Return: 
        The dictionaries of the training, dev and test set
    """
    with open(os.path.join(datadir, "train.json"), "r") as f:
        traindict = json.load(f)
    with open(os.path.join(datadir, "dev.json"), "r") as f:
        devdict = json.load(f)
    with open(os.path.join(datadir, "test.json"), "r") as f:
        testdict = json.load(f)
    return traindict, devdict, testdict


class Dataloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, feat_params):
        super(Dataloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict  # The Python-dictionary which imported from the json file
        self.window_size = feat_params[0]  # The hyper-parameters for feature extraction
        self.hop_size = feat_params[1]
        self.feature_type = feat_params[2]
        self.n_filters = feat_params[3]
        self.fbank_fmin = feat_params[4]
        self.fbank_fmax = feat_params[5]
        self.num_ceps = feat_params[6]
        self.left_context = feat_params[7]
        self.right_context = feat_params[8]
        self.data_dir = feat_params[9]

    def _get_keys(self, datadict):
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        # The audio sample file name
        filename = self.datakeys[index]
        # Get audio sample path
        filepath = self.data_dir + self.datadict[filename]["audiodir"]
        # Extract audio features by the self-programmed feature extractor
        audiofeat = fe.compute_features_with_context(filepath, self.window_size, self.hop_size, self.feature_type,
                                                     self.n_filters, self.fbank_fmin, self.fbank_fmax, self.num_ceps,
                                                     self.left_context, self.right_context)

        # Get ground-truth label
        labelpath = self.data_dir + self.datadict[filename]["targetdir"]
        sampling_rate, signal = wavfile.read(filepath)
        window_size_samples = tools.dft_window_size(self.window_size, sampling_rate)
        hop_size_samples = tools.sec_to_samples(self.hop_size, sampling_rate)
        hmm = HMM.HMM()
        label = tools.praat_file_to_target(labelpath, sampling_rate, window_size_samples,
                                           hop_size_samples, hmm)

        audiofeat = torch.FloatTensor(audiofeat)
        label = torch.FloatTensor(label) # one-hot vector
        return audiofeat, label, filename
