# ASE-Gruppe-4 

## Table of Contents
0. [GIT Command](#git-command)
1. [Aufgabe 1: Merkmalsextraktion I](#aufgabe-1)
2. [Aufgabe 2: Merkmalsextraktion II](#aufgabe-2)
3. [Aufgabe 3: Mel-Skalierte Dreiecksfilterbank](#aufgabe-3)
4. [Aufgabe 4: MFCCs](#aufgabe-4)
5. [Aufgabe 5: PyTorch Einführung](#aufgabe-5)
6. [Aufgabe 6: DNN-Training I - Vorbereitung der Daten](#aufgabe-6)

## Aufgabe 1
**Merkmalsextraktion I (Fensterung)**

![Dastellung Frames1](data/images/aufgabe1.6.png)

Figure 1: Dastellung der ersten vier Frames (mit Multiplikation mit einem
Hamming-Fenster) window_size 25ms, hop_size 10ms für TEST-MAN-AH-3O33951A.wav

---
![Dastellung Frames2](data/images/aufgabe1.7.png)

Figure 2: Dastellung der ersten vier Frames (mit Multiplikation mit einem
Hamming-Fenster) window_size 400ms, hop_size 250ms für TEST-MAN-AH-3O33951A.wav

## Aufgabe 2
**Merkmalsextraktion II (Spektralanalyse)**

![Spektogramm](data/images/aufgabe2.3.png)

Figure 3: Spektogramm für TEST-MAN-AH-3O33951A.wav

## Aufgabe 3
**Mel-Skalierte Dreiecksfilterbank**

![Dreiecksfilterbank](data/images/aufgabe3.6.png)

Figure 4: Mel Dreiecksfilterbank

---
![Mel-Spektrums](data/images/aufgabe3.7.png)

Figure 5: Mel-Spektrums für TEST-MAN-AH-3O33951A.wav

## Aufgabe 4
**MFCCs**

![MFCC_D_DD](data/images/aufgabe4.5.png)

Figure 6: MFCC_D_DD

## Aufgabe 5
**PyTorch Einführung**

`python uebung5.py --sourcedatadir .\data\VoxCeleb_gender\`
or add the parameter to the run configuration of the IDE
> if something wrong, add env variable: `KMP_DUPLICATE_LIB_OK=TRUE` to the run configuration of the IDE
> or directly run `$env:KMP_DUPLICATE_LIB_OK="TRUE"` in windows powershell

## Aufgabe 6
**DNN-Training I - Vorbereitung der Daten**

![Ground-Truth-Labels für das Beispiel TEST1](data/images/aufgabe6.5.1.png)

Figure 7: Ground-Truth-Labels für das Beispiel TEST1 TEST-WOMAN-BF-7O17O49A

![Ground-Truth-Labels für das Beispiel DEV1](data/images/aufgabe6.5.2.png)

Figure 7: Ground-Truth-Labels für das Beispiel DEV1 TEST-MAN-HJ-16O1A



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
