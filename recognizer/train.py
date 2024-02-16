import os, sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from tqdm import tqdm
from recognizer.model import *
from recognizer.utils import *
import torch.nn.functional as F

import random

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def find_best_model(filelist):
    """
    find_best_model() finds the model with best performance on the dev set.
    Input:
        filelist <list>: a list of saved model names.
    Return: 
        return the best performance on the dev set and the corresponding 
        trained model file name.
    """
    filedict = {}
    for i in filelist:
        score = float(i.split('_')[-1].strip('.pkl'))
        filedict.update({i: score})
    Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    return Keymax, filedict[Keymax]


def train(dataset, model, device, optimizer=None, criterion=None):
    """
    train() trains the model. 
    Input:
        dataset: dataset used for training.
        model: model to be trained.
        optimizer: optimizer used for training.
        criterion: defined loss function
    Return: 
        the performance on the training set and the trained model.
    """
    # bring model into training mode
    model.train()
    corrects = []
    total_loss = 0
    # traverse each batch of samples
    # add tqdm to show the progress of training
    for batch_idx, (audio_feat, label, filename) in tqdm(enumerate(dataset), total=len(dataset), desc='Train'):
        # move data onto gpu if gpu available
        audio_feat = audio_feat.to(device)
        label = label.to(device)
        label = torch.argmax(label, dim=2)  # one-hot to scalar
        # zero the parameter gradients
        optimizer.zero_grad()
        # using model compute posterior probabilities
        output = model(audio_feat)
        # swap the dim-1 and dim-2
        # output = output.transpose(1, 2)
        # compute loss value
        loss = criterion(output, label)
        total_loss += loss.item()
        # update model parameters
        loss.backward()
        optimizer.step()

        # compute accuracy
        predicted = torch.argmax(output, dim=1)
        correct = (predicted == label).sum().item() / predicted.shape[1]
        corrects.append(correct)

        # if batch_idx == 3:
        #     break

    accuracy = np.mean(corrects)
    print("\nTraining loss: ", total_loss / len(dataset))
    return accuracy, model


def evaluation(dataset, model, device, onestep=False):
    """
    evaluation() is used to evaluate the model. 
    Input:
        dataset: the dataset used for evaluation.
        model: the trained model.
    Return: 
        the accuracy on the given dataset, the predictions saved in dictionary and the model.
    """
    # bring model into evaluation mode
    model.eval()
    # traverse each batch of samples
    corrects = []
    all_preds = {}
    with torch.no_grad():
        for batch_idx, (audio_feat, label, filename) in tqdm(enumerate(dataset), total=len(dataset), desc='Ev|Te'):
            # move data onto gpu if gpu available
            audio_feat = audio_feat.to(device)
            label = label.to(device)
            label = torch.argmax(label, dim=2)  # one-hot to scalar
            output = model(audio_feat)

            # compute accuracy
            predicted = torch.argmax(output, dim=1)
            correct = (predicted == label).sum().item() / predicted.shape[1]
            corrects.append(correct)

            # compute posterior probabilities
            posterior_probs = torch.nn.functional.softmax(output, dim=1)
            check = torch.sum(posterior_probs, dim=2)
            for i in range(len(filename)):
                all_preds[filename[i]] = posterior_probs[i].cpu().numpy()

            if onestep:
                break

    accuracy = np.mean(corrects)
    return accuracy, all_preds, model


def run(config, datadicts=None):
    """
    run() trains and evaluates the model over given number of epochs.
    Input:
        config: the defined hyperparameters
        datadicts: the dictionary containing the meta-data for training, dev and test set.
    """
    traindict, devdict, testdict = datadicts
    # Parameters for feature extraction
    feat_params = [config["window_size"], config["hop_size"],
                   config["feature_type"], config["n_filters"],
                   config["fbank_fmin"], config["fbank_fmax"],
                   config["num_ceps"], config["left_context"],
                   config["right_context"], config["data_dir"]]

    # Create 3 datasets from given training, dev and test meta-data
    train_dataset = Dataloader(traindict, feat_params)
    dev_dataset = Dataloader(devdict, feat_params)
    test_dataset = Dataloader(testdict, feat_params)

    print(len(train_dataset))
    print(len(dev_dataset))
    print(len(test_dataset))

    resultsdir = config["results_dir"]
    modeldir = config["model_dir"]

    # Parameters for early stopping
    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0

    # Define loss function, model and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # feature dimension
    if config["feature_type"] == "MFCC":
        f_dim = config["num_ceps"]
    elif config["feature_type"] == "MFCC_D":
        f_dim = config["num_ceps"] * 2
    elif config["feature_type"] == "MFCC_D_DD":
        f_dim = config["num_ceps"] * 3
    else:
        f_dim = config["n_filters"]
    # context dimension
    c_dim = config["left_context"] + config["right_context"] + 1

    # input dimension
    idim = f_dim * c_dim
    # output dimension
    odim = HMM.HMM().get_num_states()

    model = Classification(idim=idim, odim=odim, hidden_dim=512)
    model = model.to(config["device"])  # move model to gpu, if gpu available
    optimizer = torch.optim.Adam(model.parameters(),  # Initialize an optimizer
                                 lr=config["lr"]
                                 )

    # Pre-loading dataset
    data_loader_train = torch.utils.data.DataLoader(train_dataset,  # Create dataset
                                                    shuffle=True,  # Randomly shuffle if shuffle=True
                                                    batch_size=config["batch_size"],  # Defined batch size
                                                    num_workers=config["NWORKER"],
                                                    # A positive integer will turn on multi-process data loading
                                                    drop_last=False)  # If drop_last=True, the data loader will drop the last batch if there are not enough remaining samples for a batch

    data_loader_dev = torch.utils.data.DataLoader(dev_dataset, shuffle=True,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"])

    model_to_save = {}
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        # Train model on training set
        trainscore, model = train(data_loader_train,
                                  model,
                                  config["device"],
                                  optimizer=optimizer,
                                  criterion=criterion)
        # Evaluate trained model on dev set
        evalscore, outpre, model = evaluation(data_loader_dev, model,
                                              config["device"], )

        print("\nEpoch: {} | Train Acc: {} | Dev Acc: {}".format(epoch, trainscore, evalscore))

        # Implementation for early stopping: If the model accuracy on the dev set does not improve in
        # 5 epochs, training is terminated.
        torch.cuda.empty_cache()
        if evalscore <= evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new eval score')
            evalacc_best = evalscore
            continuescore = continuescore + 1
            model_to_save['model'] = model
            model_to_save['epoch'] = epoch
            model_to_save['trainscore'] = trainscore
            model_to_save['evalscore'] = evalscore

        if continuescore >= run_wait:
            stop_counter = 0
        print('stop_counter', stop_counter, 'early_wait', early_wait)
        if stop_counter < early_wait:
            pass
        else:
            # save model
            for param_group in optimizer.param_groups:
                currentlr = param_group['lr']
            OUTPUT_DIR = os.path.join(modeldir,
                                      '_'.join([str(model_to_save['epoch']), str(currentlr), str(model_to_save['trainscore'])[:6],
                                                str(model_to_save['evalscore'])[:6]]) + '.pkl')
            torch.save(model_to_save['model'], OUTPUT_DIR)
            break

    # Model has trained as many epochs as specified (subject to possible early stopping).


def test(config, testdict):
    # Parameters for feature extraction
    feat_params = [config["window_size"], config["hop_size"],
                   config["feature_type"], config["n_filters"],
                   config["fbank_fmin"], config["fbank_fmax"],
                   config["num_ceps"], config["left_context"],
                   config["right_context"], config["data_dir"]]

    test_dataset = Dataloader(testdict, feat_params)

    # Now, evaluate the model on test set:
    data_loader_test = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                   batch_size=config["batch_size"],
                                                   num_workers=config["NWORKER"])

    # Finding the model with the best performance on dev set
    modeldir = config["model_dir"]
    besttrainmodel, besttrainacc = find_best_model(os.listdir(modeldir))
    # Load model
    model = torch.load(os.path.join(modeldir, besttrainmodel),
                       map_location=config["device"])

    # Finally, evaluate the trained model on the test set and save the prediction.
    testacc, outpre, _ = evaluation(data_loader_test, model, config["device"], onestep=True)

    print("\nTest Acc: {}".format(testacc))
    return outpre


def wav_to_posteriors(model, audio_file_dict, parameters):
    # get model
    model = torch.load(model, map_location=parameters["device"])

    # get audio features
    feat_params = [parameters["window_size"], parameters["hop_size"],
                   parameters["feature_type"], parameters["n_filters"],
                   parameters["fbank_fmin"], parameters["fbank_fmax"],
                   parameters["num_ceps"], parameters["left_context"],
                   parameters["right_context"], parameters["data_dir"]]

    test_dataset = Dataloader(audio_file_dict, feat_params)

    data_loader_test = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                   batch_size=parameters["batch_size"],
                                                   num_workers=parameters["NWORKER"])

    testacc, outpre, _ = evaluation(data_loader_test, model, parameters["device"], onestep=True)
    for i, (audiofeat, label, filename) in enumerate(data_loader_test):
        # plot the label of the first two samples
        if i == 0:
            print( "Test Acc of {} is {}".format(filename[0], testacc))
            # subplot 0: show the label of the first sample
            plt.subplot(2, 1, 1)
            plt.imshow(label[0].numpy().T)
            plt.gca().invert_yaxis()
            plt.title("label of " + filename[0])
            # subplot 1: show the outpre of the first sample
            plt.subplot(2, 1, 2)
            plt.imshow(outpre[filename[0]])
            plt.gca().invert_yaxis()
            plt.title("prediction of " + filename[0])

            plt.subplots_adjust(right=0.8)
            cax = plt.axes([0.85, 0.1, 0.02, 0.8])  # [x位置, y位置, 宽度, 高度]
            plt.colorbar(cax=cax)
            plt.show()
        else:
            break
    return outpre