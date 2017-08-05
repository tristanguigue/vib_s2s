import tensorflow as tf
import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


def tf_binarize(images, threshold=0.1):
    return tf.cast(threshold < images, tf.float32)


def kl_divergence_with_std(mu, sigma, sequence=False):
    if sequence:
        reduce_index = 2
    else:
        reduce_index = 1
    return 0.5 * tf.reduce_sum(
        tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, reduce_index)


def kl_divergence(mu, sigma, mu0, sigma0, sequence=False):
    if sequence:
        reduce_index = 2
    else:
        reduce_index = 1
    return 0.5 * tf.reduce_sum(
        tf.square((mu - mu0) / sigma0) + tf.square(sigma / sigma0) +
        tf.log(tf.square(sigma0)) - tf.log(tf.square(sigma)) - 1, reduce_index)


class Batcher():
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.labels = labels
        self.pointer = 0
        self.num_examples = data.shape[0]
        self.num_batches = int(self.num_examples / batch_size)

    def reset_batch_pointer(self, shuffle=True):
        self.pointer = 0
        if shuffle:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            if self.labels:
                self.labels = self.labels[perm]

    def next_batch(self):
        x = self.data[self.pointer * self.batch_size: (self.pointer + 1) * self.batch_size]
        y = None
        if self.labels is not None:
            y = self.labels[self.pointer * self.batch_size: (self.pointer + 1) * self.batch_size]
        self.pointer += 1
        return x, y


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, input_file, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x = self.x_batches[self.pointer]
        self.pointer += 1
        return x, None

    def reset_batch_pointer(self):
        self.pointer = 0