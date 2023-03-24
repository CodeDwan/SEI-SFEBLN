# -*- coding: utf-8 -*-
"""
add signal processing nodes
"""

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time
from scipy import stats, fft


def show_accuracy(predictLabel, Label): 
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))


def tansig(x):
    return (2/(1+np.exp(-2*x)))-1


def sigmoid(data):
    return 1.0/(1+np.exp(-data))
    

def linear(data):
    return data
    

def tanh(data):
    return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    

def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z


def sparse_bls(A, b):
    # 生成特征窗的W，稀疏函数
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    # L1 是 (AA+np.eye(m))的逆矩阵,(10,10)
    L1 = np.mat(AA + np.eye(m)).I
    # L2 (10,785)
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk

def real2complex(x):
    x_complex = x[..., 0:int(x.shape[-1] / 2)] + 1j * x[..., int(x.shape[-1] / 2):]
    return x_complex

def sp_mean(x):
    x = real2complex(x)
    x_mean = np.mean(x, axis=1)
    x_mean = np.expand_dims(x_mean, axis=1)
    x_mean = np.concatenate((x_mean.real, x_mean.imag), axis=1)
    return x_mean

def sp_var(x):
    x = real2complex(x)
    x_var = np.var(x, axis=1)
    x_var = np.expand_dims(x_var, axis=1)
    x_var = np.concatenate((x_var.real, x_var.imag), axis=1)
    return x_var

def sp_std(x):
    x = real2complex(x)
    x_std = np.std(x, axis=1)
    x_std = np.expand_dims(x_std, axis=1)
    x_std = np.concatenate((x_std.real, x_std.imag), axis=1)
    return x_std

def sp_ske(x):  #偏度
    x = real2complex(x)
    x_ske = stats.skew(x, axis=1)
    x_ske = np.expand_dims(x_ske, axis=1)
    x_ske = np.concatenate((x_ske.real, x_ske.imag), axis=1)
    return x_ske

def sp_kur(x):  #峰度
    x = real2complex(x)
    x_kur = stats.kurtosis(x, axis=1)
    x_kur = np.expand_dims(x_kur, axis=1)
    x_kur = np.concatenate((x_kur.real, x_kur.imag), axis=1)
    return x_kur

def sp_fft(x, n):
    # x = real2complex(x)
    x_fft = fft.fft(x, n, axis=1)
    x_fft = np.concatenate((x_fft.real, x_fft.imag), axis=1)
    return x_fft

def sp_hfft(x, n):

    x_hfft = fft.hfft(x, n, axis=1)
    x_hfft = np.concatenate((x_hfft.real, x_hfft.imag), axis=1)
    return x_hfft

def sp_dct(x, n):

    x_dct = fft.dct(x, n=n, axis=1)
    x_dct = np.concatenate((x_dct.real, x_dct.imag), axis=1)
    return x_dct




def SFEBLN(train_x, train_y, test_x, test_y, x_sp, s, c, N1, N2, N3, fftn, N_SP):
    L = 0
    # train_x (60000,784),样本6w，每个样本784点(28*28得出)
    train_x = preprocessing.scale(train_x, axis=1)
    # 横向拼接,给偏置,FeatureOfInputDataWithBias(60000*785)
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    train_acc_all = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始
    for i in range(N2):
        random.seed(i)
        # 随机生成W_ei (785,10)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        # 特征XW (60000,10)
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        # 最大最小值归一化，归一化范围(0,1)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        # 每个特征窗口的权重(785,10)
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        # 每个特征窗口输出Z_i (60000,10)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        # 总的特征窗口Z1，Z2，...Zn  (60000,100)
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

    # 生成增强节点
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

    if N1 * N2 >= N3:
        # random.seed(67797325)
        random.seed()
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        # random.seed(67797325)
        random.seed()
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)

    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成信号处理节点SP
    train_x_complex = real2complex(x_sp)
    x_fft = sp_fft(train_x_complex, n=fftn)
    x_fft = preprocessing.scale(x_fft, axis=1)
    x_dct = sp_dct(train_x_complex, n=fftn)
    x_dct = preprocessing.scale(x_dct, axis=1)
    InputOfSPLayer = np.concatenate((x_fft, x_dct), axis=1)
    OutputOfSPori = tansig(InputOfSPLayer)
    InputOfSPLayerWithBias = np.hstack(
        [InputOfSPLayer, 0.1 * np.ones((InputOfSPLayer.shape[0], 1))]
    )
    if InputOfSPLayer.shape[1] >= N_SP:
        # random.seed(67797326)
        random.seed()
        weightOfSPLayer = LA.orth(2 * random.randn(InputOfSPLayerWithBias.shape[1], N_SP)) - 1
    else:
        # random.seed(67797326)
        random.seed()
        weightOfSPLayer = LA.orth(2 * random.randn(InputOfSPLayerWithBias.shape[1], N_SP).T - 1).T
    tempOfOutputOfSPLayer = np.dot(InputOfSPLayerWithBias, weightOfSPLayer)
    parameterOfShrink_SP = s / np.max(tempOfOutputOfSPLayer)
    OutputOfSPLayer = tansig(tempOfOutputOfSPLayer * parameterOfShrink)
    # 生成最终输入
    # InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer, OutputOfSPLayer])
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer,
                                    OutputOfSPLayer])
    # 求伪逆
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput, train_y)  # 输出结果权重
    # 计算结束，终止计时
    time_end = time.time()
    trainTime = time_end - time_start
    #
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')
    train_acc_all[0][0] = trainAcc
    train_time[0][0] = trainTime
    #-----------------------------------------------------------------------------#
    # 测试过程，读入测试数据集
    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    # 开始计时，测试时间
    time_start = time.time()
    # 计算测试集的特征窗口，特征生成权重W_ei和训练时保持一致
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                    outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin

    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    # 计算测试集的SP窗口，特征生成权重W_sp和训练时保持一致

    test_x_complex = real2complex(test_x)
    x_fft = sp_fft(test_x_complex, n=fftn)
    x_fft = preprocessing.scale(x_fft, axis=1)
    x_dct = sp_dct(test_x_complex, n=fftn)
    x_dct = preprocessing.scale(x_dct, axis=1)
    InputOfSPLayerTest = np.concatenate((x_fft, x_dct), axis=1)
    OutputOfSPoriTest = tansig(InputOfSPLayerTest)
    InputOfSPLayerWithBiasTest = np.hstack(
        [InputOfSPLayerTest, 0.1 * np.ones((InputOfSPLayerTest.shape[0], 1))]
    )

    tempOfOutputOfSPLayerTest = np.dot(InputOfSPLayerWithBiasTest, weightOfSPLayer)
    OutputOfSPLayerTest = tansig(tempOfOutputOfSPLayerTest * parameterOfShrink_SP)
   
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest,
                                        OutputOfSPLayerTest])
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return test_acc, test_time, train_acc_all, train_time, OutputOfTest
