import torch
import torch.nn as nn
import os
# read ./dataset/TIDIGITS-ASE/TRAIN/TextGrid/TRAIN-MAN-AE-1A.TextGrid as json

# read all data under ./dataset/TIDIGITS-ASE/TRAIN/TextGrid/
train_data = os.listdir('dataset/TIDIGITS-ASE/TRAIN/TextGrid/')
word_dict = {}
for x in ['', 'oh', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
    word_dict[x] = 0
for i in range(len(train_data)):
    train_data[i] = './dataset/TIDIGITS-ASE/TRAIN/TextGrid/' + train_data[i]
    # item []:
    # 	item [1]:
    # 		class = "IntervalTier"
    # 		name = "words"
    # 		xmin = 0.0
    # 		xmax = 0.870375
    # 		intervals: size = 3
    # 			intervals [1]:
    # 				xmin = 0.0
    # 				xmax = 0.160
    # 				text = ""
    # 			intervals [2]:
    # 				xmin = 0.160
    # 				xmax = 0.650
    # 				text = "one"
    # 			intervals [3]:
    # 				xmin = 0.650
    # 				xmax = 0.870375
    # 				text = ""

    ## get item intervals text
    with open(train_data[i], 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'text' in line:
                word = line.split('"')[1]
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
            if 'item [2]:' in line:
                break
print(word_dict.values())
