import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, num_actions, batch_size):
        self.inference(num_actions, batch_size)

    def inference(self, num_actions, batch_size):
        self.x = tf.placeholder(shape=[batch_size, 3],
                           dtype=tf.float32)

        h_1 = tf.layers.dense(inputs=self.x,
                              units=32,
                              activation=tf.nn.relu)

        h_2 = tf.layers.dropout(inputs=h_1,
                                rate=0.2)

        h_3 = tf.layers.dense(inputs=h_2,
                              units=16,
                              activation=tf.nn.relu)

        out = [None for i in range(num_actions)]

        for i in range(num_actions):
          out[i] = tf.layers.dense(inputs=h_3,
                                   units=1,
                                   activation=tf.nn.sigmoid)

        self.actions = tf.stack(out, axis=1)
