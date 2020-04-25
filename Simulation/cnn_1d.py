import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from Simulation.simu_data import Simu_Data_read
from utility import batch_norm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 数据获取
data = Simu_Data_read()

# 选择训练与测试集
X_train = data.X_train_minmax
y_train = data.y_train
X_test = data.X_test_minmax
y_test = data.y_test

# X_test_m = data.X_test_m_minmax
# y_test_m = data.y_test_m
# X_train1 = np.vstack((X_train,X_test_m))
# y_train1 = np.vstack((y_train,y_test_m))

# 设置训练参数
epoch = 200
batch_size = 32


total_batch = X_train.shape[0] // batch_size


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#初始化单个卷积核上的偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#输入特征x，用卷积核W进行卷积运算，strides为卷积核移动步长，
#padding表示是否需要补齐边缘像素使输出图像大小不变
def conv2d(x, W, s):
    return tf.nn.conv2d(x, W, strides=[1, 1, s, 1], padding='SAME')

#对x进行最大池化操作，ksize进行池化的范围，
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],strides=[1, 1, 4, 1], padding='VALID')

# 声明输入图片数据，类别
x = tf.placeholder('float32', [None, 400])
y = tf.placeholder('float32', [None, 4])
keep_prob = tf.placeholder("float")
# 输入图片数据转化
x_image = tf.reshape(x, [-1, 1, 400, 1])

#x_image_drop = tf.nn.dropout(x_image, keep_prob)

W_conv1 = weight_variable([1, 10, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1, 1) + b_conv1,True))
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([1, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(batch_norm(conv2d(h_pool1, W_conv2, 1) + b_conv2,True))
h_pool2 = max_pool_2x2(h_conv2)

# W_conv3 = weight_variable([1, 3, 32, 64])
# b_conv3 = bias_variable([64])
# h_conv3 = tf.nn.relu(batch_norm(conv2d(h_pool2, W_conv3, 1) + b_conv3,True))
# h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([1*25*32,128])
#W_fc11 = weight_variable([10*10*64,1024])
# 偏置值
b_fc1 = bias_variable([128])
#b_fc11 = bias_variable([1024])
# 将卷积的产出展开
#h_conv2_flat = tf.reshape(h_conv2,[-1,10*10*64])
h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 32])

# 神经网络计算，并添加relu激活函数
h_fc1 = tf.nn.relu(batch_norm(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,True,False))
#h_fc11 = tf.nn.relu(batch_norm(tf.matmul(h_conv2_flat, W_fc11) + b_fc11,True,False))
#h_fc1 = tf.expand_dims(h_fc1,1)
#h_fc11 = tf.expand_dims(h_fc11,1)
#h_fc0 = tf.concat([h_fc1, h_fc11],1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128,4])
b_fc2 = bias_variable([4])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# #
# # 增加一个代价函数
# #
# m = y.shape[0].value
# E = tf.ones([batch_size,1])  # 单位向量
# F = h_fc1  # 样本特征
# Y = y   # 样本标签
# num = tf.matmul(Y,E,transpose_a=True, transpose_b=False)  # 各种标签的样本数量
# C = tf.matmul(Y,F,transpose_a=True, transpose_b=False)  # 各种标签的样本各维特征值之和
# C_avg = tf.divide(C,num)   # 各种标签的样本平均特征值
# A = tf.reduce_sum(tf.square(F - tf.matmul(Y,C_avg)),axis=1,keepdims=True)
# D = tf.matmul(Y,A,transpose_a=True, transpose_b=False)/num  # 各类方差
# S = tf.sqrt(D) # 各类标准差
# def dist(C_avg,S):
#     m = C_avg.shape[0].value
#     #dist = tf.sqrt(tf.reduce_sum(tf.square(C[m-2] - C[m-1]))) - S[m-2] - S[m-1]
#     dist = 1-tf.exp(-tf.norm(C_avg[m-2] - C_avg[m-1], ord=2)) #- 1.0*tf.norm(S[m-2] + S[m-1], ord=1)
#     for i in range(m):
#         for j in range(i+1,m):
#             #d = tf.sqrt(tf.reduce_sum(tf.square(C_avg[i] - C_avg[j]))) - S[i] - S[j]
#             d = 1-tf.exp(-tf.norm(C_avg[i] - C_avg[j],ord=2)) #- 1.0*tf.norm(S[i] + S[j],ord=1)
#             tempt = dist
#             dist = tf.cond(tf.less(d, tempt), lambda: d, lambda: dist)
#     return -dist
# #S_add = tf.add(tf.matmul(S,E,transpose_a=False, transpose_b=True) , tf.transpose(S))
# dist = dist(C_avg,S)
# # 增加一个代价函数
m = y.shape[0].value
E = tf.ones([batch_size,1])  # 单位向量
F = h_fc1  # 样本特征
Y = y   # 样本标签
num = tf.matmul(Y,E,transpose_a=True, transpose_b=False)  # 各种标签的样本数量
C = tf.matmul(Y,F,transpose_a=True, transpose_b=False)  # 各种标签的样本各维特征值之和
Cc = tf.reduce_mean(C, axis=0)
C_avg = tf.divide(C,num+1e-10)   # 各种标签的样本平均特征值
A = tf.reduce_sum(tf.square(F - tf.matmul(Y,C_avg)),axis=1,keepdims=True)
D = tf.matmul(Y,A,transpose_a=True, transpose_b=False)    # /num  # 各类方差
# S = tf.sqrt(D) # 各类标准差
Sw = tf.reduce_sum(D,axis=0)
Sb = tf.reduce_sum(tf.multiply(tf.norm(C_avg - Cc, ord=2),num))

# dist = tf.norm(C_avg[0] - C_avg[1], ord=2)
# for i in range(m):
#     for j in range(i+1,m):
#         dist = dist + tf.norm(C_avg[i] - C_avg[j],ord=2)

center_loss = (Sw / Sb) # (1-tf.exp(-Sw)) + 20*tf.exp(-Sb) # (Sw / Sb) #Sw - 2.0 * Sb

# 代价函数
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv + 1e-10),1))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv + 1e-10),1))
regularization_loss = tf.reduce_mean(tf.square(W_conv1))\
                      +tf.reduce_mean(tf.square(W_conv2))\
                      +tf.reduce_mean(tf.square(W_fc1))\
                      +tf.reduce_mean(tf.square(W_fc2))

loss = cross_entropy + 0.2 * center_loss #+ 0.0001 * dist
# 使用Adam优化算法来调整参数
optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

# 测试正确率
#correct_prediction = tf.equal(tf.round(y_pred), y)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

# 保存模型
#saver = tf.train.Saver(max_to_keep=1)
#max_acc = 0

# 所有变量进行初始化
initial = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(initial)
    for i in range(epoch):
        for batch in range(total_batch):
            batch_x = X_train[batch*batch_size : (batch+1)*batch_size, :]
            batch_y = y_train[batch*batch_size : (batch+1)*batch_size, :]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            if i % 10 == 0 and batch % 20 == 0:
                train_sw = sess.run(Sw, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                print("train sw is: %g"%(train_sw))
                train_sb= sess.run(Sb, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                print("train sb is: %g" % (train_sb))
                train_cross1 = sess.run(cross_entropy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                print("train cross is: %g" % (train_cross1))
                train_loss1 = sess.run(loss, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                print("train loss is: %g" % (train_loss1))

        # test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        #if test_acc > max_acc:
        #    max_acc = test_acc
        #    saver.save(sess, "model/my-model", global_step=i + 1)

        if i % 10 == 0:
            print("step %d:"%(i))
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: y_train, keep_prob: 1.0})
            print("training accuracy %g" % (train_accuracy))
            test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
            print("test accuracy %g" % (test_accuracy))
            # test_accuracy1 = sess.run(accuracy, feed_dict={x: X_test_m, y: y_test_m, keep_prob: 1.0})
            # print("test accuracy1 %g" % (test_accuracy1))

sess.close()






