import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_input_batch(batch_size):
    mu = np.random.uniform(0, 1, [batch_size, 3, 2])
    return mu

def create_image(peaks, height, width):
    x_inds = []
    y_inds = []
    image = np.zeros((height, width))

    for point in peaks:
        image[height - 1 - int(point[1] * height)][int(point[0] * width)] = 1

    return image

def create_batch_image(batch_peaks):
    batch_image = np.array([])

    for peaks in batch_peaks:
        image = create_image(peaks, 100, 100)
        if not np.size(batch_image):
            batch_image = np.array([image])
        else:
            batch_image = np.vstack((batch_image, [image]))

    return batch_image


# batch_peaks = create_input_batch(2)
# print(create_batch_image(batch_peaks))
