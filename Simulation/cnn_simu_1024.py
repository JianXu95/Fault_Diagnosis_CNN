import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from CWRU.data_loader import Data_read
from Simulation.simu_data import Simu_Data_read
from utility import batch_norm, plot_embedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import time


# 数据获取
data = Simu_Data_read()

# 选择一组训练与测试集
X_train = data.X_train_minmax # 临时代替
y_train = data.y_train


# 各组测试集
X_test = data.X_test_minmax
y_test = data.y_test

epoch = 50
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


# 声明输入图片数据，类别
x = tf.placeholder('float32', [None, 1024])
y = tf.placeholder('float32', [None, 4])
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
h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 4*k])
# 神经网络计算，并添加relu激活函数
h_fc1 = tf.nn.relu(batch_norm(tf.matmul(h_pool3_flat, W_fc1) + b_fc1,True,False))

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512,64])
b_fc2 = bias_variable([64])
h_fc2 = tf.nn.relu(batch_norm(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,True,False))
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([64,4])
b_fc3 = bias_variable([4])
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# # 增加一个代价函数
m = y.shape[0].value
E = tf.ones([batch_size,1])  # 单位向量
F = h_fc2  # 样本特征
Y = y   # 样本标签
num = tf.matmul(Y,E,transpose_a=True, transpose_b=False)  # 各种标签的样本数量
C = tf.matmul(Y,F,transpose_a=True, transpose_b=False)  # 各种标签的样本各维特征值之和
C_avg = tf.divide(C,num)   # 各种标签的样本平均特征值
A = tf.reduce_sum(tf.square(F - tf.matmul(Y,C_avg)),axis=1,keepdims=True)
D = tf.matmul(Y,A,transpose_a=True, transpose_b=False)    # /num  # 各类方差
S = tf.sqrt(D) # 各类标准差

center_loss = tf.reduce_sum(D,axis=0)


# 代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv + 1e-10),1))

loss = cross_entropy #+ 0.0001*center_loss
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


converter = np.array([0,1,2,3])
TRAIN_SIZE = X_train.shape[0]
TEST_SIZE = X_test.shape[0]
train_features_cnn = np.zeros((TRAIN_SIZE, 64), dtype=float)
train_labels_cnn = np.zeros(TRAIN_SIZE, dtype=int)
test_labels_cnn = np.zeros(TEST_SIZE, dtype=int)
pred_labels_cnn = np.zeros(TEST_SIZE, dtype=int)

training = 0
if training:
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
            if test_acc > max_acc:
                max_acc = test_acc
                saver.save(sess, "model/simu-model", global_step=i + 1)
            if i % 10 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: y_train, keep_prob: 1.0})
                print("epoch %d: training accuracy %g" % (i, train_accuracy))
                test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
                print("         test accuracy %g" % (test_accuracy))

    # 训练计时结束
    end = time.time()
    print("train time: %g"%(end-start))
    sess.close()

else:
    with tf.Session() as sess:
        sess.run(initial)
        #saver = tf.train.import_meta_graph('model/my-model-113.meta')
        saver.restore(sess, tf.train.latest_checkpoint("model/"))
        test_features_cnn = h_fc2.eval(feed_dict={x: X_test,keep_prob: 1.0})
        y_conv1 = sess.run(y_conv, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        y_pred1 = tf.argmax(y_conv1, 1)
        y_pred2 = y_pred1.eval(session=sess)
        for j in range(TEST_SIZE):
            test_labels_cnn[j] = np.sum(np.multiply(converter, y_test[j, :]))
            pred_labels_cnn[j] = y_pred2[j]
        test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        print("test acc %g" % (test_acc))

    sess.close()

    F_pca = PCA(n_components=20).fit_transform(test_features_cnn)
    tsne = TSNE().fit_transform(test_features_cnn,test_labels_cnn)
    fig = plot_embedding(tsne,test_labels_cnn,'Health Conditions Clustering')
    plt.show(fig)


    label = ['H','ORF','IRF','MIX']
    confmat= confusion_matrix(y_true=test_labels_cnn,y_pred=pred_labels_cnn)#输出混淆矩阵
    # print (confmat)

    confmat = 100*(confmat.T/np.sum(confmat,1)).T
    # print(confmat)

    plt.subplots()
    plt.matshow(confmat,cmap=plt.cm.Greens,alpha=0.6)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            plt.text(x=j,y=i,s='%.2f'%confmat[i,j],fontsize=12,va='center',ha='center')
    plt.colorbar()
    plt.xticks(range(4),label,fontsize=12)
    plt.yticks(range(4),label,fontsize=12)
    plt.xlabel('Predicted label',fontsize=12)
    plt.ylabel('True label',fontsize=12)
    plt.show()












