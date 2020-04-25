import os
import random
import numpy as np
from scipy.io import loadmat,savemat
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PIL import Image

class Data_read:
    def __init__(self, snr='None'):

        # mat = loadmat('..\\Datasets\\data7_7_1\\'+snr+'\\dataset1024.mat')
        mat = loadmat('..\\Datasets\\data7\\'+snr+'\\dataset1024.mat')
        self.X_train = mat['X_train']
        self.X_test = mat['X_test']
        self.y_train = self.onehot(np.array(mat['y_train'][:,0],dtype=int))
        self.y_test = self.onehot(np.array(mat['y_test'][:,0],dtype=int))
        scaler = MinMaxScaler()
        self.X_train_minmax = scaler.fit_transform(self.X_train.T).T
        self.X_test_minmax = scaler.fit_transform(self.X_test.T).T


    def onehot(self,labels):
        '''one-hot 编码'''
        n_sample = len(labels)
        n_class = max(labels) + 1
        onehot_labels = np.zeros((n_sample, n_class))
        onehot_labels[np.arange(n_sample), labels] = 1
        return onehot_labels

    #def add_noise(self,snr):



if __name__ == '__main__':
    Data_read()
