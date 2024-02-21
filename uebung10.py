import numpy as np
import argparse
import os
import torch
import recognizer.tools as tools
import recognizer.hmm as HMM
from recognizer.train import run, test, wav_to_posteriors
from recognizer.utils import *
from recognizer.train import evaluation
from tqdm import tqdm


def test_model(datadir, hmm, model, parameters, testrun=False):
    N_total, D_total, I_total, S_total = 0, 0, 0, 0
    count = 0
    # read the data
    traindict, devdict, testdict = get_data(datadir)
    outpre = test(parameters, testdict, onestep=False, model=model)
    for key in tqdm(testdict.keys(), total=len(testdict.keys()), desc='WER calculation:'):
        lab_data = datadir + '/TIDIGITS-ASE/TEST/lab/' + key + '.lab'
        # read the lab data
        with open(lab_data, 'r') as f:
            lab = f.read().splitlines()
            # concatenate the lab data
            lab = ''.join(lab).strip().split(' ')
            testdict[key]['lab'] = lab
        # get the wav data
        posteriors_dnn = outpre[key].T
        words = hmm.posteriors_to_transcription(posteriors_dnn)
        words = [w.upper() for w in words]
        N, D, I, S = tools.needlemann_wunsch(lab, words)
        N_total += N
        D_total += D
        I_total += I
        S_total += S
        count += 1
        if testrun:
            print('-' * 50)
            print('REF: ', lab)
            print('OUT: ', words)
            print('N: ', N, 'D: ', D, 'I: ', I, 'S: ', S)
            print('current Total WER: ', 100 * (D_total + I_total + S_total) / N_total)
            if count == 3:
                break
    WER = 100 * (D_total + I_total + S_total) / N_total
    return WER


def get_args():
    parser = argparse.ArgumentParser()
    # get arguments from outside
    parser.add_argument('--sourcedatadir', default='./dataset/', type=str,
                        help='Dir saves the datasource information')
    parser.add_argument('--savedir', default='./trained/', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # parse arguments
    # data directory, e.g., /media/public/TIDIGITS-ASE
    # call:
    # python uebung10.py <data/dir>
    # e.g., python uebung11.py /media/public/TIDIGITS-ASE
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.sourcedatadir
    savedir = args.savedir
    # parameters for the feature extraction
    parameters = {'window_size': 25e-3,
                  'hop_size': 10e-3,
                  'feature_type': 'MFCC_D_DD',
                  'n_filters': 40,
                  'fbank_fmin': 0,
                  'fbank_fmax': 8000,
                  'num_ceps': 13,
                  'left_context': 10,
                  'right_context': 10}
    # default HMM
    hmm = HMM.HMM()

    # ----------------------------------------------------------------------------------------------------------
    # 1.) Test mit vorgegebenen Daten
    # die Zustandswahrscheinlichkeiten passend zum HMM aus UE6
    posteriors = np.load('data/TEST-WOMAN-BF-7O17O49A.npy')

    # Transkription für die vorgegebenen Wahrscheinlichkeiten
    words = hmm.posteriors_to_transcription(posteriors)
    print('Given posteriori OUT: {}'.format(words))  # OUT: [’SEVEN’, ’OH’, ’ONE’, ’SEVEN’, ’OH’, ’FOUR’, ’NINE’]

    # ----------------------------------------------------------------------------------------------------------
    # 2.) with audio data
    # in Übung7 trainiertes DNN Model name
    # model_name = '13_0.001_0.7004_0.6619'   # baseline
    model_name = '9_0.000001_0.8392_0.7920'   # best model
    # Model Pfad
    model_dir = os.path.join(savedir, 'model', model_name + '.pkl')
    # Laden des DNNs
    # model = torch.load(model_dir)
    if torch.cuda.is_available():
        device = 'cuda'  # use GPU
    else:
        device = 'cpu'  # use CPUS
    model = torch.load(model_dir, map_location=device)

    # Beispiel wav File
    test_audio = './TIDIGITS-ASE/TEST/wav/TEST-WOMAN-BF-7O17O49A.wav'

    # Hier bitte den eigenen Erkenner laufen lassen und das Ergebnis vergleichen
    traindict, devdict, testdict = get_data(datadir)
    file = test_audio.split('/')[-1].split('.')[0]
    parameters['device'] = device
    parameters['data_dir'] = datadir
    parameters['batch_size'] = 1
    parameters['NWORKER'] = 0
    parameters['model_dir'] = os.path.join(savedir, 'model')

    ### the following code uses directly the wav_to_posteriors function, which is required in the exercise
    posteriors_dnn_dict = wav_to_posteriors(model_dir, {file: testdict[file]}, parameters)
    posteriors_dnn = posteriors_dnn_dict[file].T
    words = hmm.posteriors_to_transcription(posteriors_dnn)

    ### the following code can also do the transcription with the model, using the test function
    # outpre = test(parameters, {file: testdict[file]}, onestep=True)
    # posteriors_dnn = outpre[file].T
    # words = hmm.posteriors_to_transcription(posteriors_dnn)

    print('OUT: {}'.format(words))  # OUT: [’SEVEN’, ’OH’, ’ONE’, ’SEVEN’, ’OH’, ’FOUR’, ’NINE’]

    # ----------------------------------------------------------------------------------------------------------
    # 3) test DNN
    wer = test_model(datadir, hmm, model, parameters)
    print('--' * 40)
    print("Total WER: {}".format(wer))
