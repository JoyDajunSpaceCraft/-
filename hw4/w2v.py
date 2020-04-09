import os
import numpy as np
import pandas as pd
import argparse
from utils import *
from gensim.models import word2vec
path_prefix = './'
def train_word2vec(x):
  # 训练work2vec 的 word embedding
  #1 ) sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。
  # 后面我们会有从文件读出的例子。
  #
  # 2) size: 词向量的维度，默认值是100。
  # 这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，
  # 比如小于100M的文本语料，则使用默认值一般就可以了。
  # 如果是超大的语料，建议增大维度。
  # 3) window：即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为𝑐，
  # window越大，则和某一词较远的词也会产生上下文关系。
  # 默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。
  # 如果是小语料则这个值可以设的更小。
  # 对于一般的语料这个值推荐在[5, 10]
  # 之间。
  # 4) sg: 即我们的word2vec两个模型的选择了。
  # 如果是0， 则是CBOW模型，是1则是Skip - Gram模型，默认是0即CBOW模型。
  # 5) hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative
  # Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical
  # Softmax。默认是0即Negative
  # Sampling。
  #
  #6) negative: 即使用Negative
  # Sampling时负采样的个数，默认是5。推荐在[3, 10]
  # 之间。这个参数在我们的算法原理篇中标记为neg。
  #
  # 　7) cbow_mean: 仅用于CBOW在做投影的时候，为0，
  # 则算法中的𝑥𝑤为上下文的词向量之和，为1则为上下文的词向量的平均值。
  # 在我们的原理篇中，是按照词向量的平均值来描述的。
  # 个人比较喜欢用平均值来表示𝑥𝑤, 默认值也是1, 不推荐修改默认值。
  # 　8) min_count: 需要计算词向量的最小词频。这
  # 个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
  # 9) iter: 随机梯度下降法中迭代的最大次数，默认是5。
  # 对于大语料，可以增大这个值。
  # 10) alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为𝜂，
  # 默认是0.025。
  # 　11) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。
  # 随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha, iter一起调参，来选择合适的三个值。
  model = word2vec.Word2Vec(x, size=250,window=5, min_count=5, workers=12,iter=10,sg=1)
  return model
if __name__ == '__main__':
  print("loading training data ...")
  train_x, y = load_training_data('training_label.txt')
  train_x_no_label = load_training_data('training_nolabel.txt')

  print("loading testing data ...")
  test_x = load_testing_data('testing_data.txt')

  model =  train_word2vec(train_x + train_x_no_label + test_x)

  print("saving model ...")
  model.save(os.path.join(path_prefix, 'w2v_all.model'))


