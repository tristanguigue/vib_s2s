import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)


def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

data = binarize(mnist.test.images)

# plt.imshow(train_data[0].reshape((28, 28)), cmap='gray')
# plt.show()

image = np.concatenate(
    (data[0][:522], np.full(784 - 522, np.nan)))

plt.imshow(image.reshape((28, 28)), cmap='gray')
plt.show()

image = np.concatenate(
    (data[1][:522], np.full(784 - 522, np.nan)))

plt.imshow(image.reshape((28, 28)), cmap='gray')
plt.show()
