import os
import json
import numpy
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch_intro.local.feature_extraction as fe
import random
SEED=666
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


def padding(sequences):
    """
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>:  The length of sequences is the same as the batch size. Each element in sequences list is a tuple with
                            a length of 3, representing the audio feature sequence in index 0, labels sequence in index 1,
                            filename sequence in index 2 (same order as the output of the sub-function def __getitem__(self, index)).
    Return:
        audio_feat_sequence <torch.FloatTensor>: The padded audio feature sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The label sequence (works with batch_size >= 1).
        filename_sequence <list>: The filename sequence (works with batch_size >= 1).
    """
    audio_feat_sequence = []
    label_sequence = []
    filename_sequence = []
    # Save the audio features, ground-truth labels and filename
    # of each sample in one batch in the corresponding list
    for i in range(len(sequences)):
        audio_feat_sequence.append(sequences[i][0])
        label_sequence.append(sequences[i][1])
        filename_sequence.append(sequences[i][2])
    # Zero-padding audio features
    audio_feat_sequence = torch.nn.utils.rnn.pad_sequence(audio_feat_sequence, batch_first=True)
    # Convert label_sequence to tensor, you can use torch.nn.utils.rnn.pad_sequence() without padding function.
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return audio_feat_sequence, label_sequence, filename_sequence


class Dataloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, feat_params):
        super(Dataloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict                        # The Python-dictionary which imported from the json file
        self.window_size = feat_params[0]               # The hyper-parameters for feature extraction
        self.hop_size = feat_params[1]
        self.feature_type = feat_params[2]
        self.n_filters = feat_params[3]
        self.fbank_fmin = feat_params[4]
        self.fbank_fmax = feat_params[5]
        self.max_len = feat_params[6]

    def _get_keys(self, datadict):
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        # The audio sample file name
        filename = self.datakeys[index]
        # Get audio sample path
        filepath = self.datadict[filename]["sampledir"]
        # Get label
        label = self.datadict[filename]["label"]
        # Extract audio features by the self-programmed feature extractor
        audiofeat = fe.compute_features(filepath, self.window_size, self.hop_size, self.feature_type,
                                        self.n_filters, self.fbank_fmin, self.fbank_fmax)
        # Move numpy audio features to FloatTensor, the tensor has dimension of [s_length, 60],
        # where s_length is the sequence length.
        audiofeat = torch.FloatTensor(audiofeat)
        # Crop the audio features, if the frame length longer than defined max_len
        if audiofeat.shape[0] > self.max_len:
            audiofeat = audiofeat[:self.max_len, :]
        # Move label to FloatTensor
        label = torch.FloatTensor([label])
        return audiofeat, label, filename
