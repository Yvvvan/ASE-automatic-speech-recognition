# ASE-Gruppe-4 

## Table of Contents
0. [GIT Command](#git-command)
1. [Aufgabe 1: Merkmalsextraktion I](#aufgabe-1)
2. [Aufgabe 2: Merkmalsextraktion II](#aufgabe-2)
3. [Aufgabe 3: Mel-Skalierte Dreiecksfilterbank](#aufgabe-3)
4. [Aufgabe 4: MFCCs](#aufgabe-4)
5. [Aufgabe 5: PyTorch Einführung](#aufgabe-5)
6. [Aufgabe 6: DNN-Training I - Vorbereitung der Daten](#aufgabe-6)
7. [Aufgabe 7: DNN-Training II](#aufgabe-7)
8. [Aufgabe 8: Viterbi-Algorithmus](#aufgabe-8)
9. [Aufgabe 9: Verbundworterkennung](#aufgabe-9)
10. [Aufgabe 10: Fertigstellung des Erkenners und Bestimmung der Wortfehlerrate](#aufgabe-10)
11. [Aufgabe 11: Best Model](#aufgabe-11)

## folder structure
```bash
├── images   #images for readme
├── data
      ├── VoxCeleb_gender # will ignored by .gitignore
      ├── TEST-MAN-AH-3O33951A.wav
      ├── TEST-WOMAN-BF-7O17O49A.npy
├── dataset               
      ├── TIDIGITS-ASE    # will ignored by .gitignore
      ├── jsons           # json for TIDIGITS
├── recongnizer
      ├── __init__.py
      ├── feature_extraction.py
      ├── hmm.py
      ├── model.py
      ├── tools.py
      ├── train.py
      ├── utils.py
├── results               # the models after training aufgabe 7
├── torch_intro           # the stuff for aufgabe 5
├── uebung1.py            # aufgabe 1
├── ...                   # aufgabe x
```


## Aufgabe 1
**Merkmalsextraktion I (Fensterung)**

<div align="center">
<img src="/images/aufgabe1.6.png" />
<p>Figure 1-1: Dastellung der ersten vier Frames (mit Multiplikation mit einem
Hamming-Fenster) window_size 25ms, hop_size 10ms für TEST-MAN-AH-3O33951A.wav
</p>
</div>

---
<div align="center">
<img src="/images/aufgabe1.7.png" />
<p>Figure 1-2: Dastellung der ersten vier Frames (mit Multiplikation mit einem
Hamming-Fenster) window_size 400ms, hop_size 250ms für TEST-MAN-AH-3O33951A.wav
</p>
</div>

## Aufgabe 2
**Merkmalsextraktion II (Spektralanalyse)**

<div align="center">
<img src="/images/aufgabe2.3.png" />
<p>
Figure 2-1: Spektogramm für TEST-MAN-AH-3O33951A.wav
</p>
</div>


## Aufgabe 3
**Mel-Skalierte Dreiecksfilterbank**

<div align="center">
<img src="/images/aufgabe3.6.png" />
<p>
Figure 3-1: Mel Dreiecksfilterbank
</p>
</div>

---
<div align="center">
<img src="/images/aufgabe3.7.png" />
<p>
Figure 3-2: Mel-Spektrums für TEST-MAN-AH-3O33951A.wav
</p>
</div>

## Aufgabe 4
**MFCCs**

<div align="center">
<img src="/images/aufgabe4.5.png" />
<p>
Figure 4-1: MFCC_D_DD
</p>
</div>

## Aufgabe 5
**PyTorch Einführung**
```bash
# formal parameter:
python uebung5.py --sourcedatadir ./VoxCeleb_gender
## or add the parameter to the run configuration of the IDE
## (this is the deault arguments, if the data is not in `./VoxCeleb_gender`, change the route)
# # environment variable:
KMP_DUPLICATE_LIB_OK=TRUE
## if something wrong, add env variable: `KMP_DUPLICATE_LIB_OK=TRUE` to the run configuration of the IDE
## or directly run `$env:KMP_DUPLICATE_LIB_OK="TRUE"` in windows powershell
```
## Aufgabe 6
**DNN-Training I - Vorbereitung der Daten**
```
# environment variable:
KMP_DUPLICATE_LIB_OK=TRUE
## or directly run `$env:KMP_DUPLICATE_LIB_OK="TRUE"` in windows powershell
```
<div align="center">
<img src="/images/aufgabe6.5.1.png" />
<p>
Figure 6-1: Ground-Truth-Labels für das Beispiel TEST1 TEST-WOMAN-BF-7O17O49A
</p>
</div>

---
<div align="center">
<img src="/images/aufgabe6.5.2.png" />
<p>
Figure 6-2: Ground-Truth-Labels für das Beispiel DEV1 TEST-MAN-HJ-16O1A
</p>
</div>


## Aufgabe 7

```bash
# formal parameter:
--datasdir ./dataset/ --savedir ./trained/
# environment variable:
 KMP_DUPLICATE_LIB_OK=TRUE
```

1 How to handle the **DATA** before feeding into the DNN (the steps below are already done in the previous aufgabe 1-6 `compute_features` and `compute_features_with_context`. 
listed here only for understanding what happened and how the size of the data changed) <br>

| step  | file                      | input stuff                                      | output size                                               | note                                                    |
|-------|---------------------------|--------------------------------------------------|-----------------------------------------------------------|---------------------------------------------------------|
| 1     | audio data                |                                                  | len_audio_samples                                         | also give the sampling_rate(f_s)                        |
| 2     | signal_frames             | f_s + window_size + hop_size                     | num_frames(f_len), window_size_samples(frame length)      | framed signal   (windowed)                              |
| 3     | STFT: spectrum (fft)      | signal_frames                                    | num_frames(f_len), window_size_samples / 2                | absolute spec, the other half is the same(redundant)    |
| 4*    | mel filter                | f_s + window_size + n_filters                    | n_filters,  window_size_samples / 2                       | the triangle filters                                    |
| 4     | FBANK: mel spectrum (mel) | dot(fft, mel_filter.T)                           | num_frames(f_len), n_filters                              | the mel spectrum, should put a log at the end           |
| 5     | MFCC: Cepstrum (ceps)     | dct(log_mel)                                     | num_frames(f_len), num_ceps                               | just make a dct to the mel, and take the first num_ceps |
| 6     | MFCC_D                    | ceps, get_delta(ceps)                            | num_frames(f_len), 2*num_ceps                             | use caps to calculate the delta, output = caps + delta  |
| 7     | MFCC_D_DD                 | ceps, ...                                        | num_frames(f_len), 3*num_ceps                             |                                                         |
 | 8     | with context              | MFCC_D_DD(or other), left_context, right_context | num_frames(f_len), 3*num_ceps(f_dim), left+right+1(c_dim) | put context on the both side                            |

> in this lessons we sometime use f as frequency, sometime use f as frames, and even feature begins with f, very confusing and annoying <br>
> here: <br>
> f_s: sampling rate, sampling **frequency** <br>
> f_len: number of **frames** (not the length of a frame, but how many frames we made for this audio)<br>
> f_dim: **feature** dimension, how many features we have for each frame. In our case MFCC_D_DD 13*3=39 <br>
> c_dim: **context** dimension, we use 10, so it's 21=10+10+1 <br>

2 get **OUTPUT** from the DNN: <br>
now we have the input data for DNN with size `(f_len, f_dim, c_dim)` <br>
DataLoader will form them in batches with a specific batch_size(bs), so the input size will be `(bs, f_len, f_dim, c_dim)`
As required, the DNN will do the following steps:
```
(bs, f_len, f_dim, c_dim) 
-> (bs, f_len, idim)         # flatten:      idim = f_dim * c_dim
-> (bs, f_len, hidden_dim)   # hidden layers (we don't care)
-> (bs, f_len, odim)         # output layer: odim = classes = hmm_states(use the script) = 106
```

3 read **LABEL** from dataset:<br>
the label from the dataset, if read properly, is one-hot encoded, with size `(f_len, odim)`<br>
To calculate cross entropy, we need to convert it to the index of the max value, with size `(f_len, 1)` or `(f_len,)`<br>
which means, the label is a list of class_index.
```
# 
(f_len, odim)       # one-hot encoded: [[0,1,0],[1,0,0], [0,0,1], ...] means [1,0,2,...]
-> (f_len, 1)       # argmax at the second dimension(dim=2), change the one-hot to index
-> (bs, f_len, 1)   # batch will get from the DataLoader
```

4 **LOSS** function:<br>
the loss function is the cross entropy, which takes the output of the DNN and the label from the dataset(one-hot not ok). <br>
the output: `(bs, f_len, odim)` <br>
the label: `(bs, f_len, 1)` <br>
here, maybe need to swap the dim of the output `(bs, odim, f_len)` (I did this step directly in model, but there should be some better ways.)<br>

5 **POSTERIOR**<br>
the output: `(bs, f_len, odim)` <br>
we take one sample from the batch(and actually bs=1 here),<br>
the output: `(f_len, odim)` <br>
turn the output to a posterior, the cross entropy loss function contains the softmax, so the last step we can use the output directly. 
So we don't add softmax in model or after output during the train.<br>
but by test, the output need to be processed by softmax. (simply, cross entropy has it, so we don't add it twice.
but test no, so we add softmax to turn the numbers into probabilities.) <br>
after softmax, the numbers in each `f_len[i]` should sum to `1`. <br> 
so the number at `f_len[i]odin[j]` is the probability of the `hmm_state[j]` at `frame[i]`. <br>

6 **ACCURACY** function:<br>
the output or the label mean, at frame `f_len[i]` the probability of the output hmm_state(class) `j` is `odin[j]`. 
from aufgabe6.5, we see the images of labels. the x-axis is the `f_len`,
the y-axis is the `hmm_state`. <br>
the job of the dnn is to predict the right hmm_state for each frame for each audio. <br>
and the requirement said each batch is an audio file (bs=1).<br>
use the label `(f_len, 1)` like `[[1],[0],[2],...]` which means at frame0 is the hmm_state1, frame1 is hmm_state0, frame2 is hmm_state2, ...<br>
to compare with the output `(f_len, odim)` but first argmax agian, so the output is `(f_len, 1)` also like `[[1],[0],[2],...]`<br>
so the accuracy of one batch is the match rate of the `f_len`-many hmm_states between the output and the label. <br>
This accuracy is the accuracy of one audio file and also, because bs=1, the accuracy of one batch. <br>
the accuracy of the whole dataset(train,evaluation,test) is the mean of all accuracy from each batch (each audio file). <br>

<div align=center>
<img src="/images/aufgabe7.5.1.png" />
<p>Figure 7-1: Posterior after Epoch-1</p>
<hr>
<img src="/images/aufgabe7.5.2.png" />
<p>Figure 7-2: Posterior best model (at Epoch-13)</p>
</div>


## Aufgabe 8
*Viterbi-Algorithmus* <br>
Go through Markov-HMM Part forwardly: <br>

we have a HMM structure, with initial input. so we can calculate the probability of each end_state (where we probably are). 
```
A: the HMM structure, a transition matrix, with size (hmm_states, hmm_states) , A_ij:probability form state_i to state_j
Π：the initial input, a vector, with size (hmm_states, 1), Π_i:probability to begin at state_i
see pic on page 18
```

① the probability of states,
we can calculate the probability of a state chain (where we probably are) 
```
P(I) = P(i_1, i_2, ..., i_T) 
= bayes rule
= P(i_2, ..., i_T | i_1)* P(i_1) 
= P(i_3, ..., i_T | i_1, i_2) * P(i_2|i_1) * P(i_1)
= P(i_4, ..., i_T | i_1, i_2, i_3) * P(i_3|i_1, i_2) * P(i_2|i_1) * P(i_1) 
...... 
= P(i_T | i_1, i_2, ..., i_T-1) * P(i_T-1 | i_1, i_2, ..., i_T-2) * ... * P(i_2|i_1) * P(i_1)
= markov assumption 
* the state i_T is only related to state i_T-1, so P(i_T | i_1, i_2, ..., i_T-1) = P(i_T |i_T-1)
= P(i_T | i_T-1) * P(i_T-1 | i_T-2) * ... * P(i_2|i_1) * P(i_1)
= f(A, Π) // f is a function of A and Π on page 19
```


② the probability of the observation,
we need know a probability of the observation to given a state.  <br>
(it's raining(state), and what's the probability i read the humidity(observation) at 80%, 50%, ...)
```
O: oberservation
I: state
P(O|I) = ? // how to get
1. we train a DNN feed O, output P(I|O), and we trust it. 
   * (so we can say we kown "i read the humidity at 80%, the probability of raining is P(I|O))
2. we use bayes rule to get P(O|I) from P(I|O) 
   * (now we know "it's raining, the probability of a humidity at 80% is P(O|I))
now we get P(O|I) and we call it b(O) the Likelihood.
```

③ we introduce an input called λ, we summarize the structure of the HMM
```
λ = (A, Π, b(O)) which is the setting of the HMM
the ① changes to P(I|λ) := the state under an input λ, no big changes
the ② changes to P(O|I,λ) := the observation under an input λ and a state I, no big changes, here *
*:= (means you give a input λ, the machine should go into state I , then you make the observation O)
```

④ we need to know what we are doing
```
we are going to find out the state chain I, which we are now located in.
* use this chain I we now the word is spelt like w-a-t-e-r
so we searching I* = argmax P(I,O|λ) (notice the difference to ②in③, here we only know the input:= setting of the HMM)
now what is P(I,O|λ),
bayes says:
P(I,O|λ) = P(I|λ) * P(O|I,λ)  = ①*②
after mathematically calculation, we get:
P(I,O|λ) = f(Π ,A ,b(O))   ..... see the formula at script page 34 
I* = argmax f(Π ,A ,b(O))
```
solved.

And actually we might define the problem first and then tear apart the formula to solve it, like ④③①②. 

Now HMM-ViterbiAlgo :
```
this part is better to read the cod in uebung8.py

I = [i_1, i_2, ..., i_T] := the best state chain
Φ = max P(I,O|λ) := the probability after we found the best state chain
Ψ = previous best node at current state
...
```

## Aufgabe 9
*Verbundworterkennung* <br>
The important thing is to understand the structure of the HMM, <br>
which is shown in the script Kap.10-11 Page 32, Page 49-50. <br>
Nothing difficult to program. Pay attention to the duplicated word. <br>

<div align="center">
<img src="/images/aufgabe9.png" />
<p>
Figure 9: The transition matrix of the HMM for the word recognition
</p>
</div>

## Aufgabe 10
*Fertigstellung des Erkenners und Bestimmung der Wortfehlerrate* <br>
```bash
# formal parameter:
--sourcedatadir ./dataset/ --savedir ./trained/
# environment variable:
 KMP_DUPLICATE_LIB_OK=TRUE
```
<div align=center>
<img src="/images/aufgabe10.3.1.png" />
<p>Figure 10-1: The first 3 WER</p>
<hr>
<img src="/images/aufgabe10.3.2.png" />
<p>Figure 10-2: The Total WER</p>
</div>

## Aufgabe 11
<div align="center">
<img src="/images/aufgabe11.jpg" />
<p>
Figure 11: Best Model
</p>
</div>

---
## GIT Command
``` bash
## STAGE & SNAPSHOT
# add a file as it looks now to your next commit (stage)
git add [file]

# unstage a file while retaining the changes in working directory
git reset [file]

# commit your staged content as a new commit snapshot
git commit -m “[descriptive message]”


## SHARE & UPDATE
# fetch and merge any commits from the tracking remote branch
git pull

# Transmit local branch commits to the remote repository branch
git push 


## REWRITE HISTORY
# clear staging area, rewrite working tree from specified commit
git reset --hard [commit id]


## TEMPORARY COMMITS
# Save modified and staged changes
git stash

# write working from top of stash stack
git stash pop

# discard the changes from top of stash stack
git stash drop


## BRANCH & MERGE
# list your branches. a * will appear next to the currently active branch
git branch

# create a new branch at the current commit
git branch [branch-name]

# switch to another branch and check it out into your working directory
git checkout
git checkout [branch-name]

```
