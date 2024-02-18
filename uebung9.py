import matplotlib
matplotlib.use('TkAgg')

import recognizer.hmm as HMM
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # default HMM
    hmm = HMM.HMM()

    ## Test getTranscription
    X0 = [0, 1, 1, 2, 2, 3]
    X020 = [1, 2, 3, 3, 31, 32, 33, 34, 35, 36, 36, 1, 2, 3, 0]
    X2002 = [31, 32, 33, 34, 35, 36, 0, 1, 2, 3, 1, 2, 3, 31, 32, 33, 34, 35, 36]

    print (hmm.getTranscription(X0))     # oh
    print (hmm.getTranscription(X020))   # oh two oh
    print (hmm.getTranscription(X2002))  # two oh oh two

    statesequence = [0, 0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 31, 32, 33, 34, 35, 36, 0, 37, 38, 39, 40, 41, 42, 43, 44, 45, 0]

    words = hmm.getTranscription(statesequence)
    print(words) # ['ONE', 'TWO', 'THREE']

    # check if the transition matrix is stochastic
    print('sum each row of transition matrix: ', np.sum(np.exp(hmm.logA), axis=1))
    # check if the initial state probabilities are stochastic
    print('sum initial state probabilities: ', np.sum(np.exp(hmm.logPi)))

    plt.imshow(np.exp(hmm.logA), cmap='Reds')
    # print(hmm.logA)
    plt.xlabel('nach Zustand j')
    plt.ylabel('von Zustand i')
    plt.colorbar()
    plt.show()

