import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
# 有三個檔案，分別是training_label.txt、training_nolabel.txt、testing_data.txt
#
# training_label.txt：有label的training data(句子配上0 or 1)
#
# training_nolabel.txt：沒有label的training data(只有句子)，用來做semi-supervise learning
#
# testing_data.txt：你要判斷testing data裡面的句子是0 or 1
def load_training_data(path = 'training_label.txt'):
  # 两种获取label方式 如果是
  if 'training_label' in path:
    with open(path, 'r') as f:
      lines = f.readlines()
      lines = [lines.strip('\n').split(' ') for line in lines]
    x = [line[2:] for line in lines]
    y = [line[0] for line in lines]
    return x, y
  else:
    with open(path, 'r') as f:
      lines = f.readlines()
      x = [lines.strip('\n').split(' ') for line in lines]
    return x

def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1 # 大于0.5有恶意
    outputs[outputs < 0.5] = 0 # 小于0.5 无恶意
    correct = torch.sum(torch.eq(outputs, labels)).item() # TODO 查看torch.eq什么意思
    return correct

def load_testing_data(path='testing_data'):
  with open(path, 'r') as f:
    lines = f.readlines()
    X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
    X = [sen.split(' ') for sen in X]
  return X

