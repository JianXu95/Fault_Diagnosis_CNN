import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from CWRU.data_loader import Data_read
from utility import batch_norm, plot_embedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time


# 数据获取
data = Data_read()

# 选择一组训练与测试集
X_train = data.X_train_minmax # 临时代替
y_train = data.y_train


# 各组测试集
X_test = data.X_test_minmax
y_test = data.y_test

epoch = 201
batch_size = 64

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


# 声明输入图片数据，类别
x = tf.placeholder('float32', [None, 400])
y = tf.placeholder('float32', [None, 10])
# 输入图片数据转化
x_image = tf.reshape(x, [-1, 20, 20, 1])

k = 16

W_conv1 = weight_variable([5, 5, 1, k])
b_conv1 = bias_variable([k])
h_conv1 = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1) + b_conv1,True))
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, k, 2*k])
b_conv2 = bias_variable([2*k])
h_conv2 = tf.nn.relu(batch_norm(conv2d(h_pool1, W_conv2) + b_conv2,True))
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([5*5*2*k,512])
# 偏置值
b_fc1 = bias_variable([512])
# 将卷积的产出展开
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 2*k])
# 神经网络计算，并添加relu激活函数
h_fc1 = tf.nn.relu(batch_norm(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,True,False))

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv + 1e-10),1))

loss = cross_entropy
# 使用Adam优化算法来调整参数
optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

# 测试正确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

# 保存模型
# saver = tf.train.Saver(max_to_keep=1)
# max_acc = 0

# 所有变量进行初始化
initial = tf.global_variables_initializer()


# converter = np.array([0,1,2,3])
# TRAIN_SIZE = X_train.shape[0]
# TEST_SIZE = X_test.shape[0]
# train_features_cnn = np.zeros((TRAIN_SIZE, 512), dtype=float)
# train_labels_cnn = np.zeros(TRAIN_SIZE, dtype=int)
# test_labels_cnn = np.zeros(TEST_SIZE, dtype=int)

# 训练计时开始
start = time.time()
with tf.Session() as sess:
    sess.run(initial)
    for i in range(epoch):
        for batch in range(total_batch):
            batch_x = X_train[batch*batch_size : (batch+1)*batch_size, :]
            batch_y = y_train[batch*batch_size : (batch+1)*batch_size, :]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        # if test_acc > max_acc:
        #     max_acc = test_acc
        #     saver.save(sess, "../CWRU/model/cnn-model", global_step=i + 1)
        if i % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: y_train, keep_prob: 1.0})
            print("epoch %d: training accuracy %g" % (i, train_accuracy))
            test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
            print("         test accuracy %g" % (test_accuracy))

# 训练计时结束
end = time.time()
print("train time: %g"%(end-start))
sess.close()

"""
with tf.Session() as sess:
    sess.run(initial)
    #saver = tf.train.import_meta_graph('model/my-model-113.meta')
    saver.restore(sess, tf.train.latest_checkpoint("model/"))
    test_features_cnn = h_fc1.eval(feed_dict={x: X_test})
    for j in range(TEST_SIZE):
        test_labels_cnn[j] = np.sum(np.multiply(converter, y_test[j, :]))
    test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
    print("test acc %g" % (test_acc))
sess.close()

F_pca = PCA(n_components=50).fit_transform(test_features_cnn)
tsne = TSNE().fit_transform(test_features_cnn,test_labels_cnn)
fig = plot_embedding(tsne,test_labels_cnn,'Health Conditions Clustering')
plt.show(fig)
"""












