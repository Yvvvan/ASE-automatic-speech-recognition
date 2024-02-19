import argparse
from recognizer.train import run, test, wav_to_posteriors
from recognizer.utils import *
import torch
import os

import random
SEED=42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
def get_args():
    parser = argparse.ArgumentParser()
    # get arguments from outside
    parser.add_argument('--sourcedatadir', default='./dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--datasdir', default='./dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--savedir', default='./trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    sourcedatadir = args.sourcedatadir
    datasetdir = args.datasdir
    savedir = args.savedir

    # If GPU on device available, use GPU to train the model
    if torch.cuda.is_available() == True:
        device = 'cuda'     # use GPU
    else:
        device = 'cpu'      # use CPUS

    # Create folders to save the trained models and evaluation results
    modeldir = os.path.join(savedir, 'model')
    resultsdir = os.path.join(savedir, 'results')
    for makedir in [modeldir, resultsdir, datasetdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    # Load meta data as dictionary
    traindict, devdict, testdict = get_data(args.datasdir)

    # Configure hyperparameters
    config = {
        "NWORKER": 0,
        "device": device,
        "lr": 0.001,
        "batch_size": 1,
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
        "data_dir": "./dataset/",
        "results_dir": resultsdir,
        "model_dir": modeldir
    }

    #### here i spilt the train+evaluation and test, in run() will only train and evaluate, in test() will do test
    run(config, datadicts=[traindict, devdict, testdict]) # train and evaluate after each epoch
    test(config, testdict, onestep=False)   # only do test, return a dict of posteriors, key: filename, value: posteriors
                                            # onestep=True, only return the posteriors of the first file in the testdict

    #### this function is to show one sample of posteriors using one trained model
    # file = "TEST-WOMAN-BF-7O17O49A"
    # model_path = "trained/model/13_0.001_0.7004_0.6619.pkl"
    # wav_to_posteriors(model_path, {file: testdict[file]}, config, plot=True) # test



