from tensorflow.examples.tutorials.mnist import input_data
from networks import StochasticFeedForwardNetwork
from learners import SupervisedLossLearner
import argparse
import time

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
HIDDEN_SIZE = 1024
BOTTLENECK_SIZE = 256
NB_EPOCHS = 500
TRAIN_BATCH = 200
LEARNING_RATE = 0.0001
BETA = 0.001


def main(beta, learning_rate, train):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    input_size = mnist.train.images.shape[1]
    output_size = mnist.train.labels.shape[1]

    sfnn = StochasticFeedForwardNetwork(input_size, HIDDEN_SIZE, BOTTLENECK_SIZE, output_size)
    learner = SupervisedLossLearner(sfnn, beta, learning_rate, TRAIN_BATCH)
    epoch_batches = int(mnist.train.num_examples / TRAIN_BATCH)
    best_accuracy = 0

    for epoch in range(NB_EPOCHS):
        print('\nEpoch:', epoch)
        start = time.time()

        total_loss = 0
        for i in range(epoch_batches):
            batch_xs, batch_ys = mnist.train.next_batch(TRAIN_BATCH)
            total_loss += learner.train_network(batch_xs, batch_ys)

        train_accuracy = learner.test_network(mnist.train.images, mnist.train.labels)
        test_accuracy = learner.test_network(mnist.test.images, mnist.test.labels)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / epoch_batches)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print('Best accuracy')

    learner.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--beta', metavar='int', type=float, const=BETA, nargs='?', default=BETA,
        help='the value of beta, mutual information regulariser')
    parser.add_argument(
        '--rate', metavar='int', type=float, const=LEARNING_RATE, nargs='?', default=LEARNING_RATE,
        help='the learning rate for the Adam optimiser')

    args = parser.parse_args()
    main(args.beta, args.rate, train=True)
