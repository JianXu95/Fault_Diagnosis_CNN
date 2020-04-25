import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    X,Y = data[:,0], data[:,1]
    fig = plt.figure()
    #color = {0:'r',1:'b',2:'g',3:'y'}
    #for x,y,l in zip(X, Y, label):
    plt.scatter(X, Y, s=1, c=label,cmap=plt.cm.get_cmap('jet',7))
    plt.xticks([])
    plt.yticks([])
    # plt.legend(label)
    plt.colorbar(ticks=range(0,7)) #['BF','IF','OF','H']
    plt.clim(-0.5, 6.5)
    #plt.title(title)
    return fig