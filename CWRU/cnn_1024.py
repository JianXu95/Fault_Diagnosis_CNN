import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from CWRU.data_loader import Data_read
from utility import batch_norm, plot_embedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals import joblib
import time


# 数据获取
data = Data_read(snr='-4db')
# data1 = Data_read('3db')
# data2 = Data_read('0db')
# data3 = Data_read('-3db')
# 选择一组训练与测试集
X_train = data.X_train_minmax # 临时代替
y_train = data.y_train

# X_train = np.vstack((data.X_train_minmax,data1.X_train_minmax))
# y_train = np.vstack((data.y_train,data1.y_train))
# X_train = np.vstack((X_train,data2.X_train_minmax))
# y_train = np.vstack((y_train,data2.y_train))
# X_train = np.vstack((X_train,data3.X_train_minmax))
# y_train = np.vstack((y_train,data3.y_train))
#
# index = list(range(X_train.shape[0]))
# random.Random(8).shuffle(index)
# X_train = X_train[index]
# y_train = y_train[index]

# 各组测试集
# X_test = data.X_test_minmax
# y_test = data.y_test
X_test = np.vstack((data.X_test_minmax,data.X_train_minmax))
y_test = np.vstack((data.y_test,data.y_train))
print(y_test.shape)

epoch = 100
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
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#对x进行最大池化操作，ksize进行池化的范围，
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')


# def load_model():
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph('model/my-model-113.meta')
#         saver.restore(sess, tf.train.latest_checkpoint("model/"))

c = 7

# 声明输入图片数据，类别
x = tf.placeholder('float32', [None, 1024])
y = tf.placeholder('float32', [None, c])
# 输入图片数据转化
x_image = tf.reshape(x, [-1, 32, 32, 1])

k = 16


W_conv1 = weight_variable([5, 5, 1, k])
b_conv1 = bias_variable([k])
h_conv1 = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1) + b_conv1,True))
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, k, 2*k])
b_conv2 = bias_variable([2*k])
h_conv2 = tf.nn.relu(batch_norm(conv2d(h_pool1, W_conv2) + b_conv2,True))
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([2, 2, 2*k, 4*k])
b_conv3 = bias_variable([4*k])
h_conv3 = tf.nn.relu(batch_norm(conv2d(h_pool2, W_conv3) + b_conv3,True))
h_pool3 = max_pool_2x2(h_conv3)


W_fc1 = weight_variable([4*4*4*k,512])
# 偏置值
b_fc1 = bias_variable([512])
# 将卷积的产出展开
h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*4*k])
# 神经网络计算，并添加relu激活函数
h_fc1 = tf.nn.relu(batch_norm(tf.matmul(h_pool3_flat, W_fc1) + b_fc1,True,False))

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512,128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(batch_norm(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,True,False))
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([128,c])
b_fc3 = bias_variable([c])
h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
y_conv = tf.nn.softmax(h_fc3)

regularization_loss = tf.reduce_mean(tf.square(W_conv1))\
                      +tf.reduce_mean(tf.square(W_conv2))\
                      +tf.reduce_mean(tf.square(W_conv3))\
                      +tf.reduce_mean(tf.square(W_fc1))\
                      +tf.reduce_mean(tf.square(W_fc2))\
                      +tf.reduce_mean(tf.square(W_fc3))

m = y.shape[0].value
E = tf.ones([batch_size,1])  # 单位向量
F = h_fc2 #h_fc3  # 样本特征
Y = y   # 样本标签
num = tf.matmul(Y,E,transpose_a=True, transpose_b=False)  # 各种标签的样本数量
C = tf.matmul(Y,F,transpose_a=True, transpose_b=False)  # 各种标签的样本各维特征值之和
Cc = tf.reduce_mean(C, axis=0)
C_avg = tf.divide(C,num+1e-10)   # 各种标签的样本平均特征值
A = tf.reduce_sum(tf.square(F - tf.matmul(Y,C_avg)),axis=1,keepdims=True)
D = tf.matmul(Y,A,transpose_a=True, transpose_b=False)  #/  num  # 各类方差
S = tf.sqrt(D) # 各类标准差
Sw = tf.reduce_sum(D,axis=0)
# Sb = tf.reduce_sum(tf.multiply(tf.square(C_avg - Cc),num))
Sb = tf.reduce_sum(tf.multiply(tf.norm(C_avg - Cc, ord=2),num))


# 代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv + 1e-10),1))

loss = cross_entropy
# 使用Adam优化算法来调整参数
optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

# 测试正确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

# 保存模型
saver = tf.train.Saver(max_to_keep=1)
max_acc = 0

# 所有变量进行初始化
initial = tf.global_variables_initializer()


converter = np.array([0,1,2,3,4,5,6])
TRAIN_SIZE = X_train.shape[0]
TEST_SIZE = X_test.shape[0]
# train_features_cnn = np.zeros((TRAIN_SIZE, 512), dtype=float)
train_labels_cnn = np.zeros(TRAIN_SIZE, dtype=int)
test_labels_cnn = np.zeros(TEST_SIZE, dtype=int)

# 训练计时开始
training = 0
save_model = 1
eps = 0.0015
stop_n = 0
if training:
    train_acc = []
    test_acc = []
    test_accuracy = 0
    start = time.time()
    with tf.Session() as sess:
        sess.run(initial)
        for i in range(1, epoch+1):
            for batch in range(total_batch):
                batch_x = X_train[batch*batch_size : (batch+1)*batch_size, :]
                batch_y = y_train[batch*batch_size : (batch+1)*batch_size, :]
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                if i % 10 == 0 and batch % 30 == 0:
                    train_loss1 = sess.run(loss, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    print("train loss is: %g" % (train_loss1))

            test_acc1 = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
            if save_model:
                if test_acc1 >= max_acc:
                    max_acc = test_acc1
                    saver.save(sess, "../CWRU/model_7/cwt-model", global_step=i)
            # train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: y_train, keep_prob: 1.0})
            # test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
            # train_loss = sess.run(loss, feed_dict={x: X_train, y: y_train, keep_prob: 1.0})
            # test_loss = sess.run(loss, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
            # train_acc.append(train_accuracy)
            # test_acc.append(test_accuracy)
            # train_loss.append(train_loss)
            # test_loss.append(test_loss)

            if i % 5 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: y_train, keep_prob: 1.0})
                print("epoch %d: training accuracy %g" % (i, train_accuracy))
                test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
                print("epoch %d: test accuracy %g" % (i, test_accuracy))
                print("time: %g" % (time.time() - start))

    # 训练计时结束
    end = time.time()
    print("train time: %g"%(end-start))
    sess.close()
    #
    # # 存数据
    # file= open('../CWRU/model2/train_acc.txt', 'w')
    # for v in train_acc:
    #     file.write(str(v))
    #     file.write('\n')
    # file.close()
    #
    # file= open('../CWRU/model2/test_acc.txt', 'w')
    # for v in test_acc:
    #     file.write(str(v))
    #     file.write('\n')
    # file.close()
else:
    with tf.Session() as sess:
        sess.run(initial)
        #saver = tf.train.import_meta_graph('model/my-model-113.meta')new-loss-train-200
        saver.restore(sess, "../CWRU/model_7/cnn-plain-199")
        # saver.restore(sess, tf.train.latest_checkpoint("../CWRU/model22/"))
        test_features_cnn = h_fc2.eval(feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        for j in range(TEST_SIZE):
            test_labels_cnn[j] = np.sum(np.multiply(converter, y_test[j, :]))
        test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        print("test acc %g" % (test_acc))

    sess.close()

    F_pca = PCA(n_components=50).fit_transform(test_features_cnn)
    tsne = TSNE().fit_transform(F_pca,test_labels_cnn)
    fig = plot_embedding(tsne,test_labels_cnn,'Health Conditions Clustering')
    plt.show(fig)
    #











