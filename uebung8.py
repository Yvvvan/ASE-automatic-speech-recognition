import numpy as np


# Bitte diese Funktion selber implementieren
def viterbi( logLike, logPi, logA ):
    """
    Viterbi-Algorithm for HMM
    :param logLike:  log(likelihood) of the observations  b(o_t)
    :param logPi:    log(initial state)
    :param logA:     log(transition matrix)
    :return:
    stateSequence:   most likely state sequence [i_1, i_2, ..., i_T]
    pStar: p*(o|λ), max. probability of the most likely state sequence
    """
    # T: number of observations, N: number of states
    T, N = logLike.shape

    # phi[t, i] = max. probability of the most likely path ending in state i at time t
    phi = np.zeros((T, N))
    # psi[t, i] = the best previous state(argmax) at time t-1 [i_t-1] that bring us to this state i at time t
    psi = np.zeros((T, N), dtype=int)

    phi[0, :] = logPi + logLike[0, :]  # initial probability at time t=0   # multiply in log domain = add
    psi[0, :] = -1                     # no previous state at time t=0

    for t in range(1, T):   # time
        for j in range(N):  # state(nodes)
            phi[t, j] = np.max(phi[t - 1, :] + logA[:, j]) + logLike[t, j]
            psi[t, j] = np.argmax(phi[t - 1, :] + logA[:, j])

    # p*: the max. probability of the most likely path ending in state i at final time T
    # in coding, because T is length of the time array, so T-1 is the last time
    pStar = np.max(phi[T - 1, :])
    # i*: the state sequence that gives the p*
    stateSequence = np.zeros(T, dtype=int)
    # so last state is the state with the highest probability
    stateSequence[T - 1] = np.argmax(phi[T - 1, :])
    # so data saved in psi[t,j] shows the previous state of j, j is the last state we found in the last line
    for t in range(T - 2, -1, -1):  # backtracking
        # the state at time t = the saved *previous state* at time t+1 = the data saved in psi[t+1,j]
        stateSequence[t] = psi[t + 1, stateSequence[t + 1]]
    return stateSequence, pStar


def limLog(x):
    """
    Log of x.

    :param x: numpy array.
    :return: log of x.
    """
    MINLOG = 1e-100
    return np.log(np.maximum(x, MINLOG))



if __name__ == "__main__":
    # Vektor der initialen Zustandswahrscheinlichkeiten
    logPi = limLog([ 0.9, .0, 0.1 ])

    # Matrix der Zustandsübergangswahrscheinlichkeiten
    logA  = limLog([
      [ 0.8,  .0, 0.2 ], 
      [ 0.4, 0.4, 0.2 ], 
      [ 0.3, 0.2, 0.5 ] 
    ]) 

    # Beobachtungswahrscheinlichkeiten für "Regen", "Sonne", "Schnee" 
    # B = [
    #     {  2: 0.1,  3: 0.1,  4: 0.2,  5: 0.5,  8: 0.1 },
    #     { -1: 0.1,  1: 0.1,  8: 0.2, 10: 0.2, 15: 0.4 },
    #     { -3: 0.2, -2: 0.0, -1: 0.8,  0: 0.0 }
    # ]




    # gemessene Temperaturen (Beobachtungssequenz): [ 2, -1, 8, 8 ]
    # ergibt folgende Zustands-log-Likelihoods
    logLike = limLog([
      [ 0.1,  .0,  .0 ],
      [  .0, 0.1, 0.8 ],
      [ 0.1, 0.2,  .0 ],
      [ 0.1, 0.2,  .0 ]
    ])

    # erwartetes Ergebnis: [0, 2, 1, 1], -9.985131541576637
    print( viterbi( logLike, logPi, logA ) )


    # verlängern der Beobachtungssequenz um eine weitere Beobachung 
    # mit der gemessenen Temperatur 4
    # neue Beobachtungssequenz: [ 2, -1, 8, 8, 4 ]
    logLike = np.vstack( ( logLike, limLog([ 0.2, 0, 0 ]) ) )

    # erwartetes Ergebnis: [0, 2, 0, 0, 0], -12.105395077776727
    print( viterbi( logLike, logPi, logA ) )
