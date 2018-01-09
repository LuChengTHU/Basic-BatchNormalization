# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.training import moving_averages

class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss

        input = self.x_
        is_BN = True
        HH = 50

        W_fc1 = weight_variable([784, HH])
        b_fc1 = bias_variable([HH])
        h_fc1 = tf.matmul(input, W_fc1) + b_fc1

        with tf.variable_scope('bn1'):
            h_bn1 = batch_normalization_layer(h_fc1, isTrain=is_train, isBN=is_BN)

        h_relu1 = tf.nn.relu(h_bn1)

        h_relu1_drop = tf.nn.dropout(h_relu1, self.keep_prob)

        W_fc2 = weight_variable([HH, 10])
        b_fc2 = bias_variable([10])
        h_fc2 = tf.matmul(h_relu1_drop, W_fc2) + b_fc2

        #        the 10-class prediction output is named as "logits"
        logits = h_fc2

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)



def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain=True, isBN=True):
    '''
    Here I referred this website:
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    '''

    if not isBN:
        return inputs

    axis = list(range(len(inputs.get_shape()) - 1))

    mean, variance = tf.nn.moments(inputs, axis)

    beta = tf.get_variable('beta', initializer=tf.zeros_initializer(), shape=inputs.get_shape()[-1], dtype=tf.float32)
    gamma = tf.get_variable('gamma', initializer=tf.ones_initializer(), shape=inputs.get_shape()[-1], dtype=tf.float32)

    moving_mean = tf.get_variable(name='moving_mean', initializer=tf.zeros_initializer(),
                                  shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
    moving_variance = tf.get_variable(name='moving_var', initializer=tf.zeros_initializer(),
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



