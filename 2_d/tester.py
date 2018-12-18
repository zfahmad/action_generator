import tensorflow as tf
from model import Model
import numpy as np
import reward


def create_input_batch(batch_size):
    mu = np.random.uniform(0, 1, [batch_size, 16, 2])
    # mu = np.random.normal([[0.25, 0.25], [0.75, 0.75]], 0.1, [batch_size, 2, 2])
    return mu


def pairwise_distance(a, b):
    na = tf.reduce_sum(tf.square(a), axis=2)
    na = tf.transpose(tf.expand_dims(na, 1), perm=[0, 2, 1])

    # Create unit vector of points in tensor b.
    nb = tf.reduce_sum(tf.square(b), axis=2)
    nb = tf.expand_dims(nb, 1)

    return na - 2 * tf.matmul(a, tf.transpose(b, perm=[0, 2, 1])) + nb


def gauss_kernel(a, b, sigma=0.05):
    d = pairwise_distance(a, b)
    return (1 / (sigma * np.sqrt(2 * np.pi))) \
           * tf.exp(-(1 / (2 * sigma ** 2)) * d)


def objective(inputs, outputs):

    outcomes, values = tf.py_func(reward.batch_evaluate, [inputs, outputs], [tf.double, tf.double])
    values = tf.transpose(tf.expand_dims(values, 1), perm=[0, 2, 1])

    K = gauss_kernel(outputs, tf.cast(outcomes, tf.float32), sigma=0.01)
    K_norm = tf.transpose(tf.expand_dims(tf.reduce_sum(K, axis=2), 1),
                          perm=[0, 2, 1])

    return tf.divide(tf.matmul(K, tf.cast(values, tf.float32)), K_norm)

def calc_loss(r, a):
    l = tf.reduce_sum(r, axis=1)
    K = gauss_kernel(a, a, sigma=0.01)

    reg = tf.reduce_sum(K, axis=[1, 2])

    loss = tf.reduce_mean(l) - 0.01 * tf.reduce_mean(reg)
    return loss


BATCH_SIZE = 9
NUM_ACTIONS = 16
# OUTPUT_FILE = 'p16_n0-05_reg_16.dat'


m = Model(num_actions=NUM_ACTIONS, batch_size=BATCH_SIZE)
r = objective(m.x, m.actions)
loss = calc_loss(r, m.actions)
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(-loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './no_sigmoid.ckpt')

    peaks = create_input_batch(BATCH_SIZE)
    actions = sess.run(m.actions, feed_dict={m.x: peaks})

for i in range(len(peaks)):
    # print(peaks[i], actions[i])
    f = np.random.uniform(0, 4, NUM_ACTIONS)
    reward.plot_reward_space(peaks[i], f, 0, actions[i])
    #     ts, train_loss = sess.run([train_step, loss], feed_dict={m.x: peaks})
    #
    #     if not i % INTERVAL:
    #         peaks = create_input_batch(BATCH_SIZE)
    #         actions, rewards = sess.run([m.actions, loss], feed_dict={m.x: peaks})
    #         stats = reward.batch_regret(peaks, actions)
    #         print_list = [trial, i, rewards, stats[0]]
    #         print("{:5} {:5} {:10.5} {:8.5}".format(*print_list))
    #         regret[i // INTERVAL] = stats
