import tensorflow as tf
from model import Model
import numpy as np
import reward
import one_ply as op


def create_input_batch(mu_, batch_size):
    # mu = np.random.normal(mu_, 0.1, [batch_size, 3])
    mu = np.random.uniform(0, 1, [batch_size, 3])
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


BATCH_SIZE = 32
NUM_ACTIONS = 3
NUM_TRIALS = 5
TRAINING_STEPS = 10000
INTERVAL = 250
OUTPUT_FILE = 'reg_kr_np.dat'


m = Model(num_actions=NUM_ACTIONS, batch_size=BATCH_SIZE)
r = objective(m.actions, m.x)
loss = calc_loss(r, m.actions)
optimizer = tf.train.AdamOptimizer(1e-3)
train_step = optimizer.minimize(-loss)


total_regret = np.zeros([TRAINING_STEPS // INTERVAL, 2])
out_file = open(OUTPUT_FILE, 'wb')

for trial in range(NUM_TRIALS):
    regret = np.zeros([TRAINING_STEPS // INTERVAL, 2])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        mu_ = [0.25, 0.5, 0.75]
        for i in range(TRAINING_STEPS):
            peaks = create_input_batch(mu_, BATCH_SIZE)
            ts, train_loss = sess.run([train_step, loss], feed_dict={m.x: peaks})
            if not i % INTERVAL:
                peaks = create_input_batch(mu_, BATCH_SIZE)
                actions = sess.run(m.actions, feed_dict={m.x: peaks})
                stats = op.test_net(peaks, actions)
                print_list = [trial, i, train_loss, stats[0]]
                print("Trial: {:2} Step: {:5} Reward: {:10.5} Regret: {:8.5}".format(*print_list))
                regret[i // INTERVAL] = stats

        total_regret += regret

total_regret /= NUM_TRIALS
np.save(out_file, total_regret)
