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
                                   units=2)

        self.actions = tf.stack(out, axis=1)
