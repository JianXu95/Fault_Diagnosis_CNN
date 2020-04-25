import os
import random
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split


class Simu_Data:
    def __init__(self, length, m, FE=0):
        # root directory of all data
        rdir = os.path.join('..\Datasets\simulation')
        all_lines = open('..\\Datasets\\simulation\\metadata_new.txt').readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            lines.append(l)

        self.length = length  # sequence length
        self.samples = m  # sample number
        self._slice_data(rdir, lines, FE)
        # shuffle training and test arrays
        # self._shuffle()
        self.labels = tuple(line[1] for line in lines)
        self.nclasses = len(self.labels)  # number of classes

        """
        self.X_test_m = np.zeros((0, self.length))
        self.y_test_m = np.zeros((0, 3))
        mat_dict = loadmat('Datasets\\simulation\\fault12.mat')
        time_series = mat_dict['X4'][0,:]
        samples = self.samples
        for sample in range(samples):
            start = sample * self.length // 2
            segment = time_series[start: start + self.length]
            self.X_test_m = np.vstack((self.X_test_m, segment))
            self.y_test_m = np.vstack((self.y_test_m, np.array([0,1,1])))
        """

    def _slice_data(self, rdir, infos, FE):
        # self.X_train = np.zeros((0, self.length))
        # self.X_test = np.zeros((0, self.length))
        # self.y_train = []
        # self.y_test = []
        if not FE:
            self.X = np.zeros((0, self.length))
            self.y = np.zeros((0, 1))
        else:
            self.X = np.zeros((0, 15))
            self.y = np.zeros((0, 1))
        for idx, info in enumerate(infos):
            # directory of this file
            fpath = os.path.join(rdir, info[1] + '.mat')
            mat_dict = loadmat(fpath)
            key = info[0]
            time_series = mat_dict[key][0,:]
            samples = self.samples
            for sample in range(samples):
                start = (7 * sample * self.length) // 10
                segment = time_series[start: start + self.length]
                if not FE:
                    feat = segment
                else:
                    feat = self.feat_extract(segment)
                self.X = np.vstack((self.X, feat))
                self.y = np.vstack((self.y, idx))

            # idx_last = -(time_series.shape[0] % self.length)
            # 原始信号长度除以单个样本信号长度的余数，截掉不用
            # clips = time_series[:idx_last].reshape(-1, self.length)
            # n = clips.shape[0]
            # 获得特定信号长度的样本数
            # 按3:1切分训练和测试集
            # n_split = int((3 * n / 4))
            # self.X_train = np.vstack((self.X_train, clips[:n_split]))
            # self.X_test = np.vstack((self.X_test, clips[n_split:]))
            # self.y_train += [idx] * n_split
            # self.y_test += [idx] * (clips.shape[0] - n_split)
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.X, self.y, test_size=0.4, random_state=10)

    def feat_extract(self, x):
        feat = pd.DataFrame(index=[0])
        feat['mean'] = np.mean(x)
        feat['std'] = np.std(x)
        feat['max'] = np.max(x)
        feat['min'] = np.min(x)
        feat['peak'] = np.max(np.abs(x))
        feat['p2p'] = np.max(x) - np.min(x)
        feat['rms'] = np.sqrt(np.mean(np.square(x)))
        feat['smr'] = np.square(np.mean(np.sqrt(abs(x))))
        feat['ma'] = np.mean(abs(x))
        feat['kurt'] = stats.kurtosis(x)
        feat['skew'] = stats.skew(x)
        feat['shape_f'] = feat['rms'] / feat['ma']
        feat['crest_f'] = feat['peak'] / feat['rms']
        feat['impulse_f'] = feat['peak'] / feat['ma']
        feat['clear_f'] = feat['peak'] / feat['smr']

        return feat



"""
    def _shuffle(self):
        # shuffle training samples

        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = tuple(self.y_train[i] for i in index)

        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = tuple(self.y_test[i] for i in index)
"""

from sklearn.preprocessing import MinMaxScaler

class Simu_Data_read:
    def __init__(self):

        mat = loadmat('..\\Datasets\\simulation\\dataset1024.mat')
        self.X = mat['X']
        self.X_train = mat['X_train']
        self.X_test = mat['X_test']
        self.y_train = self.onehot(np.array(mat['y_train'][:,0],dtype=int))
        self.y_test = self.onehot(np.array(mat['y_test'][:,0],dtype=int))
        self.y = self.onehot(np.array(mat['y'][:, 0], dtype=int))
        scaler = MinMaxScaler()
        self.X_minmax = scaler.fit_transform(self.X.T).T
        self.X_train_minmax = scaler.fit_transform(self.X_train.T).T
        self.X_test_minmax = scaler.fit_transform(self.X_test.T).T
        # self.X_test_m = mat['X_test_m']
        # self.X_test_m_minmax = scaler.fit_transform(self.X_test_m.T).T
        # self.y_test_m = mat['y_test_m']
        # 1D -- 2D
        #self.X_train_2D = self.X_train_minmax.reshape(-1,20,20)
        #self.X_test_2D = self.X_test_minmax.reshape(-1,20,20)

    def onehot(self,labels):
        '''one-hot 编码'''
        n_sample = len(labels)
        n_class = max(labels) + 1
        onehot_labels = np.zeros((n_sample, n_class))
        onehot_labels[np.arange(n_sample), labels] = 1
        return onehot_labels


## 处理获得用于学习训练的数据，存为mat文件方便下次使用
if __name__ == '__main__':
    data = Simu_Data(length=1024, m=250, FE=0)
    savemat('..\\Datasets\\simulation\\dataset1024.mat', {'X_train': data.X_train,
                                                'X_test': data.X_test,
                                                'y_train': data.y_train,
                                                'y_test': data.y_test,
                                                'X':data.X,
                                                'y':data.y})
                                                #'X_test_m':data.X_test_m,
                                                #'y_test_m':data.y_test_m})




