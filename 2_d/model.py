import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, num_actions, batch_size):
        self.inference(num_actions, batch_size)

    def inference(self, num_actions, batch_size):
        self.x = tf.placeholder(shape=[batch_size, 16, 2],
                           dtype=tf.float32)

        h_1 = tf.layers.flatten(self.x)

        h_2 = tf.layers.dense(inputs=h_1,
                              units=64,
                              activation=tf.nn.relu)

        h_3 = tf.layers.dense(inputs=h_2,
                              units=32,
                              activation=tf.nn.relu)

        out = [None for i in range(num_actions)]

        for i in range(num_actions):
          out[i] = tf.layers.dense(inputs=h_3,
                                   units=2,
                                   activation=tf.nn.sigmoid)

        self.actions = tf.stack(out, axis=1)


class CNNModel:
    def __init__(self, num_actions, batch_size):
        self.inference(num_actions, batch_size)

    def inference(self, num_actions, batch_size):
        self.peaks = tf.placeholder(shape=[batch_size, 16, 2],
                           dtype=tf.float32)

        self.x = tf.placeholder(shape=[batch_size, 100, 100],
                           dtype=tf.float32)

        img = tf.expand_dims(self.x, 3)

        h_1 = tf.layers.conv2d(img, 2, 8, 1)

        h_1_pool = tf.layers.average_pooling2d(h_1, 8, 4)

        h_2 = tf.layers.conv2d(h_1_pool, 4, 8, 1)

        h_2_pool = tf.layers.average_pooling2d(h_2, 8, 4)

        h_3 = tf.layers.flatten(h_2_pool)

        h_4 = tf.layers.dense(inputs=h_3,
                              units=64,
                              activation=tf.nn.relu)

        h_5 = tf.layers.dense(inputs=h_4,
                              units=32,
                              activation=tf.nn.relu)

        out = [None for i in range(num_actions)]

        for i in range(num_actions):
          out[i] = tf.layers.dense(inputs=h_5,
                                   units=2,
                                   activation=tf.nn.sigmoid)

        self.actions = tf.stack(out, axis=1)
