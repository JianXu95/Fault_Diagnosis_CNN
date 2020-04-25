from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from CWRU.data_loader import Data_read
from scipy.io import loadmat,savemat
from utility import batch_norm, plot_embedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# 数据获取

mat = loadmat('Datasets\\data\\10\\dataset1024_15.mat')
X_train = mat['X_train']
X_test = mat['X_test']
y_train = np.array(mat['y_train'][:,0])
y_test = np.array(mat['y_test'][:,0])
# y_train = mat['y_train']
# y_test = mat['y_test']
# print(X_train[:10])


clf1 = SVC(C=1, kernel='rbf') #
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)

Accuracy1 = accuracy_score(y_test, y_pred1)
# F1_score1 = f1_score(y_test, y_pred1, average='macro')
print("Test Accuracy of SVM is %f %%"%(Accuracy1*100))
# print("Test F1_score of SVM is %d %%"%(F1_score1*100))

clf2 = MLPClassifier(hidden_layer_sizes=(30,), batch_size=64, learning_rate_init=0.05, max_iter=400)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

Accuracy2 = accuracy_score(y_test, y_pred2)
# F1_score2 = f1_score(y_test, y_pred2, average='macro')
print("Test Accuracy of BPNN is %f %%"%(Accuracy2*100))
# print("Test F1_score of BPNN is %d %%"%(F1_score2*100))

clf3 = KNeighborsClassifier(n_neighbors=5)
clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)

Accuracy3 = accuracy_score(y_test, y_pred3)
# F1_score3 = f1_score(y_test, y_pred3, average='macro')
print("Test Accuracy of kNN is %f %%"%(Accuracy3*100))
# print("Test F1_score of kNN is %d %%"%(F1_score3*100))