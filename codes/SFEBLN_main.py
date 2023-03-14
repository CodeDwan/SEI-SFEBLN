# -*- coding: utf-8 -*-
import numpy as np
import h5py
from SFEBLN import *
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
from scipy import fftpack
from utils_SFEBLN import *


if __name__ == "__main__":
    # 参数初始化
    picflag = False
    pic_path = './Modelsave_BLS/figure/C10.jpg'
    datapath = '../Datasets/ADS-B_10.mat'

    # 数据处理，导入，打包
    print('----------------External Data Processing----------------')
    x_train, y_train, x_test, y_test = readdata_longADSB(datapath)
    x_train_fft = sp_fft(real2complex(x_train), n=2048)
    x_test_fft = sp_fft(real2complex(x_test), n=2048)
    x_sp = x_train

    # x_train, y_train, x_test, y_test = few_shot(50, x_train, y_train, x_test, y_test)
    print(f"X train: {x_train.shape}, Y train: {y_train.shape}")
    print(f"X test: {x_test.shape}, Y_test: {y_test.shape}")

    # 宽度学习参数
    N1 = 20  #  # of nodes belong to each window
    N2 = 10  #  # of windows -------Feature mapping layer
    N3 = 3000 #  # of enhancement nodes -----Enhance layer
    s = 0.5  #  shrink coefficient
    C = 2**-24 # Regularization coefficient

    print('-------------------SFEBLN----------------------')
    fftn = 32
    N4 = 100
    _, _, _, _, outputlabel = SFEBLN(x_train_fft, y_train, x_test_fft, y_test, x_sp, s, C, N1, N2, N3, fftn, N4)

    # plot confusion matrix
    if picflag:
        print('-------------------Plot Confusion Matrix------------------------')
        y_test = np.argmax(y_test, axis=1)
        outputlabel = np.argmax(outputlabel, axis=1)
        cm_plot(y_test, outputlabel, pic=pic_path)



