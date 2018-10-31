import tensorflow as tf
from model import Model
import numpy as np
import reward


def create_input_batch(mu_, batch_size):
    mu = np.random.normal(mu_, 0.01, [batch_size, 3])
    return mu


def pairwise_distance(a, b):
    na = tf.reduce_sum(tf.square(a), axis=2)
    na = tf.transpose(tf.expand_dims(na, 1), perm=[0, 2, 1])
    nb = tf.reduce_sum(tf.square(b), axis=2)
    nb = tf.expand_dims(nb, 1)
    return na - 2 * tf.matmul(a, tf.transpose(b, perm=[0, 2, 1])) + nb


def gauss_kernel(a, b, sigma=0.05):
    d = pairwise_distance(a, b)
    return (1 / (sigma * np.sqrt(2 * np.pi))) \
           * tf.exp(-(1 / (2 * sigma ** 2)) * d)


def objective(a, peaks):

    outcomes, values = tf.py_func(reward.evaluate, [a, peaks], 2 * [tf.float32])

    K = gauss_kernel(a, outcomes)
    K_norm = tf.transpose(tf.expand_dims(tf.reduce_sum(K, axis=2), 1), perm=[0, 2, 1])

    return tf.divide(tf.matmul(K, values), K_norm)


def calc_loss(r, a):
    l = tf.reduce_sum(r, axis=1)
    K = gauss_kernel(a, a)
    K_norm = tf.transpose(tf.expand_dims(tf.reduce_sum(K, axis=2), 1), perm=[0, 2, 1])

    reg = tf.reduce_sum(K, axis=[1, 2])

    loss = tf.reduce_mean(l) - 1 * tf.reduce_mean(reg)
    return loss


BATCH_SIZE = 10
NUM_ACTIONS = 3

m = Model(num_actions=NUM_ACTIONS, batch_size=BATCH_SIZE)
r = objective(m.actions, m.x)

loss = calc_loss(r, m.actions)

optimizer = tf.train.AdamOptimizer(1e-3)
train_step = optimizer.minimize(-loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    mu_ = [0.25, 0.5, 0.75]
    for i in range(5000):
        peaks = create_input_batch(mu_, BATCH_SIZE)
        ts, t_loss, actions, rw = sess.run([train_step, loss, m.actions, r], feed_dict={m.x: peaks})
        if not i % 100:
            print(t_loss)
            print(actions)
