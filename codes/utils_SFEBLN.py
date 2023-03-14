import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
import h5py
import math
from sklearn.manifold import TSNE
import pandas as pd
from scipy import stats, fft

def addwgn(x, snr):
    x_noise = np.zeros(shape=x.shape, dtype=float)
    for i in range(len(x)):
        signal = x[i]
        # signal_power = np.mean(np.power(np.abs(signal),2))
        signal_power = np.linalg.norm(signal) ** 2 / signal.size
        noise = np.random.randn(signal.size)  # 产生N(0,1)噪声数据
        noise = noise - np.mean(noise)
        noise_variance = signal_power / np.power(10, (snr / 10))
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
        noise = np.reshape(noise, signal.shape)
        signal_noise = signal+noise
        x_noise[i] = signal_noise
    return x_noise

def cm_plot(original_label, predict_label, pic=None):
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵

    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    if pic is not None:
        plt.savefig(pic)
    plt.show()

def one_hot(labels):
    one_hot = np.asarray(pd.get_dummies(labels))
    #print("One-hot Labels shape:", one_hot.shape)
    #print("One-hot Labels:", '\n', one_hot)
    return one_hot

def split(X, Y, ratio):
    #split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, stratify=Y)
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=424)
    # for train_index, test_index in sss.split(X, Y):
    #     x_train, x_test = X[train_index], X[test_index]
    #     y_train, y_test = Y[train_index], Y[test_index]
    return x_train, y_train, x_test, y_test


def real2complex(x):
    x_complex = x[..., 0:int(x.shape[-1] / 2)] + 1j * x[..., int(x.shape[-1] / 2):]
    return x_complex

def sp_fft(x, n):
    # x = real2complex(x)
    x_fft = fft.fft(x, n, axis=1)
    x_fft = np.concatenate((x_fft.real, x_fft.imag), axis=1)
    return x_fft



def readdata_longADSB(datapath):
    hdict = h5py.File(datapath, 'r')
    X_temp = np.transpose(hdict['data'][:])
    Y_temp = np.transpose(hdict['label'][:])
    X = np.concatenate((X_temp[..., 0], X_temp[..., 1]), axis=1)
    Y = one_hot(Y_temp.reshape(len(Y_temp)))
    x_train, y_train, x_test, y_test = split(X, Y, 0.3)

    return x_train, y_train, x_test, y_test

