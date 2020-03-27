import numpy as np
import pandas as pd

def dataprocess(df):
  # 处理数据 完成
  x_list = []
  y_list = []
  # 替换空值 转换数字形式
  df = df.replace(['NR'], [0.0])
  array = np.array(df).astype(float)
  for i in range(0, 4320, 18):
    for j in range(15):
      mat = array[i:i + 18, j:j + 9]
      label = array[i + 9, j + 9]
      x_list.append(mat)
      y_list.append(label)
  x = np.array(x_list)
  y = np.array(y_list)
  # x.shape(3600, 18, 9)
  # y.shape(3600,)

  return x, y, array


def train(x_train, y_train, epoch):
  bias = 0
  learning_rate = 1
  weights = np.ones(9)
  reg_rate = 0.0001
  bg2_sum = 0
  wg2_sum = np.zeros(9)
  for i in range(epoch):
    w_g = np.zeros(9)
    b_g = 0
    for j in range(3200):
      b_g += (-1) * (y_train[j] - weights.dot(x_train[j, 9, :]) - bias)
      for k in range(9):
        w_g[k] += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) *(-x_train[j, 9, k])
    b_g /= 3200
    w_g /= 3200

    for m in range(9):
      w_g[m] += reg_rate * weights[m]

    bg2_sum += b_g ** 2
    wg2_sum += w_g ** 2  # 注意形状
    bias -= learning_rate / bg2_sum ** 0.5 * b_g
    weights -= learning_rate / wg2_sum ** 0.5 * w_g
    # 每训练200轮，输出一次在训练集上的损失
    if i % 200 == 0:
      loss = 0
      for j in range(3200):
        loss += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) ** 2
      print('after {} epochs, the loss on train data is:'.format(i), loss / 3200)
  return weights, bias



def vaildation(x_val, y_val, weights, bias):
  loss = 0
  for i in range(400):
    loss += (y_val[i] - weights.dot(x_val[i, 9, :]) - bias) ** 2
  return loss / 400


def main():
  df = pd.read_csv("train(1).csv", usecols=range(3, 27), encoding='utf-8')
  x, y, _ = dataprocess(df)
  x_train, y_train = x[0:3200], y[0:3200]
  epoch = 2000
  w, b = train(x_train, y_train, epoch)
  print(w, b)
  x_val, y_val = x[3200:3600], y[3200:3600]
  # loss = vaildation(x_val, y_val, w, b)
  # print('The loss on val data is:', loss)


if __name__ == '__main__':
  main()
