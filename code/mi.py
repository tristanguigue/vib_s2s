from tensorflow.examples.tutorials.mnist import input_data
from networks import StochasticFeedForwardNetwork
from learners import SupervisedLossLearner
import argparse
import time
from tools import Batcher

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
HIDDEN_SIZE = 1024
BOTTLENECK_SIZE = 256
NB_EPOCHS = 500
BATCH_SIZE = 500
LEARNING_RATE = 0.0001
BETA = 0.001


def main(beta, learning_rate, train):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    input_size = mnist.train.images.shape[1]
    output_size = mnist.train.labels.shape[1]

    sfnn = StochasticFeedForwardNetwork(input_size, HIDDEN_SIZE, BOTTLENECK_SIZE, output_size, True)
    learner = SupervisedLossLearner(sfnn, beta, learning_rate, BATCH_SIZE)
    epoch_batches = int(mnist.train.num_examples / BATCH_SIZE)
    train_loader = Batcher(mnist.train.images, mnist.train.labels, BATCH_SIZE)
    test_loader = Batcher(mnist.train.images, mnist.test.labels, BATCH_SIZE)
    best_accuracy = 0

    for epoch in range(NB_EPOCHS):
        print('\nEpoch:', epoch)
        start = time.time()

        total_loss = 0
        for i in range(epoch_batches):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            total_loss += learner.train_network(batch_xs, batch_ys, LEARNING_RATE)

        train_loss, train_accuracy = learner.test_network(train_loader)
        test_loss, test_accuracy = learner.test_network(test_loader)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / epoch_batches)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)
        print('Train loss: ', train_loss, ', test loss: ', test_loss)

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
