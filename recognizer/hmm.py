import numpy as np
from recognizer.tools import viterbi

# default HMM
WORDS = {
    'name': ['sil', 'oh', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
    'size': [1, 3, 15, 12, 6, 9, 9, 9, 12, 15, 6, 9],
    'gram': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    # 'gram': [19561, 2568, 2594, 2593, 2586, 2577, 2582, 2545, 2596, 2576, 2567, 2545],
}


class HMM:

    words = {}
    logPi = np.array([])  # initial state probabilities: Size = num_states * 1
    logA = np.array([])  # state transition probabilities : Size = num_states * num_states

    def __init__(self, words=WORDS):
        """
        Constructor of HMM class. Inits with provided structure words
        :param input: word of the defined HMM.
        """
        self.words = words
        num_states = self.get_num_states()
        num_words = len(words['name'])
        # Initialize logPi to all zeros and then take the log
        self.logPi = np.zeros(num_states)
        self.logPi[0] = 1.0  # Assuming the first state is the starting state
        # set zero to a very small value
        self.logPi[self.logPi == 0] = 1e-10
        # log
        self.logPi = np.log(self.logPi)

        # Initialize logA with the transitions probabilities and then take the log
        self.logA = np.zeros((num_states, num_states))

        # Assuming each state has a transition to itself and the next state
        # Linear HMM
        for i, size in enumerate(WORDS['size']):
            start_index = sum(WORDS['size'][:i])
            end_index = start_index + size
            # Transition intra word
            for j in range(start_index, end_index):
                self.logA[j, j] = 1          # Transition to itself
                if j + 1 < end_index:
                    self.logA[j, j + 1] = 1  # Transition to next state

            # Transition inter word (all other words)
            for k in range(num_words):
                start_index_k = sum(WORDS['size'][:k])
                self.logA[end_index - 1, start_index_k] = 1

        # normalize each row of the transition matrix
        self.logA = self.logA / self.logA.sum(axis=1)[:, None]
        # set zero to a very small value
        self.logA[self.logA == 0] = 1e-10

        # log
        self.logA = np.log(self.logA)



    def get_num_states(self):
        """
        Returns the total number of states of the defined HMM.
        :return: number of states.
        """
        return sum(self.words['size'])

    def input_to_state(self, input):
        """
        Returns the state sequenze for a word.
        :param input: word of the defined HMM.
        :return: states of the word as a sequence.
        """
        if input not in self.words['name']:
            raise Exception('Undefined word/phone: {}'.format(input))

        # start index of each word
        start_idx = np.insert(np.cumsum(self.words['size']), 0, 0)

        # returns index for input's last state
        idx = self.words['name'].index(input) + 1

        start_state = start_idx[idx - 1]
        end_state = start_idx[idx]

        return [n for n in range(start_state, end_state) ]


    def getTranscription(self, stateSequence):
        """
        Returns the word for a state sequence.
        :param stateSequence: state sequence of the word.
        :return: word of the state sequence.
        """
        transcription = []
        word_start_state = {self.words['name'][i]: sum(self.words['size'][:i]) for i in range(len(self.words['name']))}
        word_end_state = {self.words['name'][i]: sum(self.words['size'][:i + 1]) for i in range(len(self.words['name']))}
        last_state = None

        for state in stateSequence:
            for word in self.words['name']:
                if word_start_state[word] <= state < word_end_state[word]:
                    # ignore the silence
                    if word != 'sil':
                        # ignore the repeated words (intra word)
                        if len(transcription) == 0 or transcription[-1] != word:
                            transcription.append(word)
                        # dont ignore the inter repeated words
                        elif len(transcription) > 0 and state < last_state and transcription[-1] == word:
                            transcription.append(word)
                    break
            last_state = state
        transcription = [word.upper() for word in transcription]
        # return ' '.join(transcription)     # return string
        return transcription                 # return list

    def posteriors_to_transcription(self, posteriors):
        """
        Returns the transcription words
        """
        # log likelihood
        # 1. replace 0 with a small value
        posteriors = np.where(posteriors == 0, 1e-10, posteriors)
        # 2. calculate the log likelihood
        posteriors = np.log(posteriors)
        state_sequence, pStar = viterbi(posteriors, self.logPi, self.logA)
        return self.getTranscription(state_sequence)









""
