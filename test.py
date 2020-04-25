import numpy as np
import pandas as pd
import heapq
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat,savemat

t = [0,0.1,0.5,0.7,0.9,0.2,1.0,0.4,-0.1,-0.7,-0.2]
re = map(t.index, heapq.nlargest(3,t))
tt = np.zeros(8)
for i in list(re):
    tt[i] = t[i]
s = pd.Series(t).apply(lambda x: 1 if x>0.5 else 0)



def sparse_process(x):
    length = len(x)
    xx = np.zeros(length)
    for i in range(length // 5):
        seg = abs(np.array(x[i * 5: (i + 1) * 5]))
        idx = map(list(seg).index, heapq.nlargest(3, seg))
        for j in list(idx):
            xx[i * 5 + j] = seg[j]
    return xx

# data = loadmat('Datasets\\data\\10\\dataset1024.mat')
# c = 93
# img = data['X_train'][c]
# label = data['y_train'][c]
# print(label)
# plt.imshow(img.reshape(32,32),cmap='Greys')
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# plt.show()

index = ['C1C2P1','C1P1C2P2','C1C2P1C3P2','C1P1C2P2C3P3']
acc_4 = [98.85,99.43,99.10,99.18]
acc_8 = [99.10,99.25,99.30,99.55]
acc_16 = [99.17,99.28,99.15,99.83]
acc_20 = [99.72,99.45,99.55,99.75]
time_4 = [458.05,319.74,328.74,144.20]
time_8 = [565.62,188.77,443.16,198.47]
time_16 = [472.79,247.20,312.22,358.29]
time_20 = [321.80,313.73,437.52,473.52]

ind = np.arange(4)+0.6
ind2 = [i-0.1 for i in ind]
ind1 = [i-0.3 for i in ind]
ind3 = [i+0.1 for i in ind]
ind4 = [i+0.3 for i in ind]
bar_width = 0.2
plt.figure(figsize=(18, 4))
plt.bar(ind1 , acc_4, width=0.18 , alpha=1, color='orange',label = "4 filters")
plt.bar(ind2 , acc_8, width=0.18 , alpha=1,color='limegreen',label = "8 filters")
plt.bar(ind3 , acc_16, width=0.18 , alpha=1,color='deeppink',label = "16 filters")
plt.bar(ind4 , acc_20, width=0.18 , alpha=1,color='dodgerblue',label = "20 filters")
plt.legend(loc='upper')
plt.ylabel('Accuracy (%)',size=14)
plt.xlabel('Models',size=14)
plt.ylim(98,100)
plt.xticks(ind,index,size=12)
plt.yticks(size=12)
plt.show()
