import tensorflow as tf
from abc import ABC, abstractmethod
from tools import kl_divergence_with_std


class Learner(ABC):
    def __init__(self, network, learning_rate, train_batch):
        self.lr = tf.placeholder(tf.float32)
        self.net = network
        self.learning_rate = learning_rate
        self.train_batch = train_batch
        self.loss_op = self.loss()

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss_op)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def loss(self):
        pass

    def train_network(self, batch_xs, batch_ys, learning_rate):
        if learning_rate:
            self.learning_rate = learning_rate
        feed_dict = {
            self.net.x: batch_xs,
            self.lr: self.learning_rate
        }
        if batch_ys is not None:
            feed_dict.update({self.net.y_true: batch_ys})

        _, current_loss = self.sess.run([self.train_step, self.loss_op], feed_dict=feed_dict)

        return current_loss

    def next_batch(self, items, i):
        return items[i * self.train_batch: (i + 1) * self.train_batch]

    def test_network(self, data, labels):
        total_accuracy = 0
        nb_batches = data.shape[0] // self.train_batch
        for i in range(nb_batches):
            feed_dict = {self.net.x: self.next_batch(data, i)}
            if labels is not None:
                feed_dict.update({self.net.y_true: self.next_batch(labels, i)})
            batch_accuracy = self.sess.run(self.net.accuracy, feed_dict=feed_dict)
            total_accuracy += batch_accuracy

        return total_accuracy / nb_batches

    def test_network_loader(self, loader):
        total_accuracy = 0
        nb_batches = loader.num_batches
        loader.reset_batch_pointer()
        for i in range(nb_batches):
            batch_xs, batch_ys = loader.next_batch()
            feed_dict = {self.net.x: batch_xs}
            if batch_ys:
                feed_dict.update({self.net.y_true: batch_ys})
            batch_accuracy = self.sess.run(self.net.accuracy, feed_dict=feed_dict)
            total_accuracy += batch_accuracy

        return total_accuracy / nb_batches


class SupervisedLossLearner(Learner):
    def __init__(self, network, beta, learning_rate, train_batch):
        self.beta = beta
        super().__init__(network, learning_rate, train_batch)

    def loss(self):
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.net.y_true, logits=self.net.decoder_output)

        kl_loss = kl_divergence_with_std(self.net.mu, self.net.sigma)

        if self.beta:
            return tf.reduce_mean(cross_entropy_loss + self.beta * kl_loss)
        return tf.reduce_mean(cross_entropy_loss)


class PredictionLossLearner(Learner):
    def __init__(self, network, beta, learning_rate, train_batch):
        self.beta = beta
        super().__init__(network, learning_rate, train_batch)

    def loss(self):
        true_pixels = self.net.inputs[:, 1:]
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_pixels, logits=self.net.decoder_output[:, :-1])

        kl = kl_divergence_with_std(self.net.mu, self.net.sigma)
        kl = tf.reshape(kl, [-1, self.net.seq_size, self.net.output_size])

        if self.beta:
            return tf.reduce_mean(cross_entropy + self.beta * kl[:, :-1])
        return tf.reduce_mean(cross_entropy)


class CharPredictionLossLearner(Learner):
    def __init__(self, network, beta, learning_rate, train_batch):
        self.beta = beta
        super().__init__(network, learning_rate, train_batch)

    def loss(self):
        true_char = tf.one_hot(indices=self.net.x[:, 1:], depth=self.net.output_size)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_char, logits=self.net.decoder_output[:, :-1])

        kl = kl_divergence_with_std(self.net.mu, self.net.sigma)
        kl = tf.reshape(kl, [-1, self.net.seq_size])

        if self.beta:
            return tf.reduce_mean(cross_entropy) + self.beta * tf.reduce_mean(kl[:, :-1])
        return tf.reduce_mean(cross_entropy)
