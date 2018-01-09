# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.training import moving_averages

class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.00001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        x = tf.reshape(self.x_, [-1, 28, 28, 1])

        # TODO: implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss

        channel_size = 8
        kernel_size = 5
        isBN = True

        W_conv1 = weight_variable([kernel_size, kernel_size, 1, channel_size])
        b_conv1 = bias_variable([channel_size])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        h_bn1 = batch_normalization_layer(h_conv1, 'bn1', isTrain=is_train, isBN=isBN)

        relu1 = tf.nn.relu(h_bn1)

        h_pool1 = max_pool(relu1, 2)

        W_conv2 = weight_variable([kernel_size, kernel_size, channel_size, channel_size])
        b_conv2 = bias_variable([channel_size])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        h_bn2 = batch_normalization_layer(h_conv2, 'bn2', isTrain=is_train, isBN=isBN)

        relu2 = tf.nn.relu(h_bn2)

        h_pool2 = max_pool(relu2, 2)

        # reorder so the channels are in the first dimension, x and y follow.
        layer2 = tf.transpose(h_pool2, (0, 3, 1, 2))
        layer2_flat = tf.reshape(layer2, [-1, 7 * 7 * channel_size])

        W_fc1 = weight_variable([7 * 7 * channel_size, 10])
        b_fc1 = bias_variable([10])
        h_fc1 = tf.matmul(layer2_flat, W_fc1) + b_fc1

        #        the 10-class prediction output is named as "logits"
        logits = h_fc1

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, pool_size):
  return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, pool_size, pool_size, 1], padding='SAME')


def batch_normalization_layer(inputs, name, isTrain=True, isBN=True):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    
    '''
    Here I referred this website:
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    '''

    if not isBN:
        return inputs

    with tf.variable_scope(name):
        axis = list(range(len(inputs.get_shape()) - 1))

        mean, variance = tf.nn.moments(inputs, axis)

        beta = tf.get_variable('beta', initializer=tf.zeros_initializer, shape=inputs.get_shape()[-1], dtype=tf.float32)
        gamma = tf.get_variable('gamma', initializer=tf.ones_initializer, shape=inputs.get_shape()[-1], dtype=tf.float32)

        moving_mean = tf.get_variable(name='moving_mean', initializer=tf.zeros_initializer,
                                      shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
        moving_variance = tf.get_variable(name='moving_var', initializer=tf.zeros_initializer,
                                          shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.999)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.999)

        if isTrain:
            tf.add_to_collection('mean', update_moving_mean)
            tf.add_to_collection('variance', update_moving_variance)
        else:
            mean = update_moving_mean
            variance = update_moving_variance

        inputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)

        return inputs
