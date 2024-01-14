import argparse
import matplotlib
matplotlib.use('TkAgg')

from recognizer import utils
from torch_intro.local.train import run
import torch
import os
import matplotlib.pyplot as plt
import random

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


if __name__ == '__main__':
    # If GPU on device available, use GPU to train the model
    if torch.cuda.is_available():
        device = 'cuda'  # use GPU
    else:
        device = 'cpu'  # use CPUS

    # Configure hyperparameters
    config = {
        "NWORKER": 0,
        "device": device,
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 50,
        "window_size": 25e-3,
        "hop_size": 10e-3,
        "feature_type": "MFCC_D_DD",
        "n_filters": 40,
        "fbank_fmin": 0,
        "fbank_fmax": 8000,
        "num_ceps": 13,
        "left_context": 10,
        "right_context": 10,
        "data_dir": "./data",
    }

    feat_params = [config["window_size"], config["hop_size"],
                   config["feature_type"], config["n_filters"],
                   config["fbank_fmin"], config["fbank_fmax"],
                   config["num_ceps"], config["left_context"],
                   config["right_context"], config["data_dir"]]

    """¨Uberpr¨ufung und Darstellung der Ergebnisse: Erstellen Sie ein uebung6.py-Skript,
    in dem Sie einen Data-Loader f¨ur das Development-Set erstellen. Setzen Sie die
    Batchgr¨oße bitte auf 1. Iterieren Sie durch Ihren Data-Loader, um sicherzustellen,
    dass dieser die extrahierten Merkmale und die entsprechenden Ground-Truth-Labels
    zur¨uckgibt. Plotten Sie die Ground-Truth-Labels der ersten beiden Samples, die vom
    Data-Loader zur¨uckgegeben wurden, und ¨uberpr¨ufen Sie, ob sie unterschiedlich sind.
    Falls sie gleich aussehen, k¨onnte Ihr Data-Loader einen Fehler aufweisen. Setzen Sie
    sampling rate=16000."""

    # Load data
    traindict, devdict, testdict = utils.get_data('./dataset')
    # Create data loader
    dev_dataset = utils.Dataloader(devdict, feat_params)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (audiofeat, label, filename) in enumerate(dev_loader):
        print(audiofeat.shape)
        print(label.shape)
        print(filename)
        # plot the label of the first two samples
        if i < 2:
            #用matplotlib plot labelmatrix
            plt.figure()
            plt.imshow(label[0].numpy().T)
            plt.gca().invert_yaxis()
            plt.title('label of file: <dev> ' + filename[0])
            plt.show()
        if i == 2:
            break




