import tensorflow as tf
from model import CNNModel
import numpy as np
import reward
import image as img


def create_input_batch(batch_size):
    mu = np.random.uniform(0, 1, [batch_size, 16, 2])
    # mu = np.random.normal([[0.25, 0.25], [0.75, 0.75]], 0.1, [batch_size, 2, 2])
    return mu


BATCH_SIZE = 9
NUM_ACTIONS = 16

m = CNNModel(num_actions=NUM_ACTIONS, batch_size=BATCH_SIZE)

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./models/img_p16_n0-05_reg_16_gk0-05_r0-01.ckpt")

    peaks = create_input_batch(BATCH_SIZE)
    images = img.create_batch_image(peaks)
    actions = sess.run(m.actions, feed_dict={m.x: images, m.peaks: peaks})

for i in range(len(peaks)):
    f = np.random.uniform(0, 4, 16)
    reward.plot_reward_space(peaks[i], f, 0, i, actions[i])
