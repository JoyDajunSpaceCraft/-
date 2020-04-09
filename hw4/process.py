from torch import nn
from gensim.models import Word2Vec
import torch


# data 预处理
class Preprocess():
  def __init__(self, sentences, sen_len, w2v_path='./w2v.model'):
    self.w2v_path = w2v_path
    self.sentences = sentences
    self.sen_len = sen_len
    self.idx2word = []
    self.word2idx = {}
    self.embedding_matrix = []

  def get_w2v_model(self):
    # 读入word to vector 模型
    self.embedding = Word2Vec.load(self.w2v_path)
    self.embedding_dim = self.embedding.vector_size

  def add_embedding(self, word):
    # 把word加进embedding，赋予随机的representation vector
    # word 只会是<PAD> 或者 <UNK>

    vector = torch.empty(1, self.embedding_dim)
    torch.nn.init.uniform_(vector)  # 均匀分布 https://pytorch-cn.readthedocs.io/zh/latest/package_references/nn_init/

    # 对Word进行空间映射
    self.word2idx[word] = len(self.word2idx)
    self.idx2word.append(word)
    self.embedding_matrix = torch.cat([self.embedding_matrix, vector],
                                      0)  # https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchcat
    print("加入 PAD 和 UNK 之后的Word{}".format(self.word2idx))

  def make_embedding(self, load=True):
    print("Get embedding ...")
    # 取得训练好的 embedding
    if load:
      print("loading word to vec model ...")
      self.get_w2v_model()
    else:
      raise NotImplementedError
    for i, word in enumerate(self.embedding.wv.vocab):  # model.wv.vocab：可以直接调用生成的词向量
      print('get words #{}'.format(i + 1), end='\r')
      # e.g. self.word2index['魯'] = 1
      # e.g. self.index2word[1] = '魯'
      # e.g. self.vectors[1] = '魯' vector
      self.word2idx[word] = len(self.word2idx)
      self.idx2word.append(word)
      self.embedding_matrix.append(self.embedding(word))
    print("word embedding done!")
    self.embedding_matrix = torch.tensor(self.embedding_matrix)

    # 加入 <PAD> 和 <UNK>
    self.add_embedding("<PAD>")
    self.add_embedding("<UNK>")
    print("total words: {}".format(len(self.embedding_matrix)))
    return self.embedding_matrix

  def pad_sequence(self, sentence):
    """
    将每个句子保持一样长度
    :param sentence: word2idx
    :return:
    """
    if len(sentence) > self.sen_len:
      sentence = sentence[:self.sen_len]
    else:
      pad_len = self.sen_len - len(sentence)
      for _ in pad_len:
        sentence.append(self.word2idx['<PAD>'])

    assert len(sentence) == self.sen_len
    return sentence


  def sentence_word2idx(self):
    """
    将句子转换成对应的index
    :return:
    """
    sentence_list = []
    for i, sen in enumerate(self.sentences):
      print("sentence count {}".format(i+1), end='\r')
      sentence_idx = []
      for word in sen:
        if word in self.word2idx.keys():
          sentence_idx.append(self.word2idx[word])
        else:
          sentence_idx.append(self.word2idx['<UNK>'])
      sentence_idx = self.pad_sequence(sentence_idx)
      sentence_list.append(sentence_idx)
    return torch.LongTensor(sentence_list)


  def labels_to_tensor(self, y):
    """
    将label转换为tensor
    :param y:
    :return:
    """
    y = [int(label) for label in y]
    return torch.LongTensor(y)