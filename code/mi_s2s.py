from tensorflow.examples.tutorials.mnist import input_data
from networks import Seq2Seq
from learners import PartialPredictionLossLearner
from tools import Batcher
import argparse
import time

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
HIDDEN_SIZE = 1024
BOTTLENECK_SIZE = 256
NB_EPOCHS = 500
BATCH_SIZE = 200
LEARNING_RATE = 0.001
BETA = 0.001
OUTPUT_SEQ_SIZE = 100
OUTPUT_SIZE = 1


def main(beta, learning_rate, layers):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    seq_size = mnist.train.images.shape[1]
    partial_sequence_size = int(2 * seq_size / 3)

    train_loader = Batcher(mnist.train.images, None, BATCH_SIZE)
    test_loader = Batcher(mnist.test.images, None, BATCH_SIZE)

    seq2seq = Seq2Seq(seq_size, partial_sequence_size, OUTPUT_SEQ_SIZE, HIDDEN_SIZE,
                      BOTTLENECK_SIZE, OUTPUT_SIZE, layers, True)
    learner = PartialPredictionLossLearner(seq2seq, beta, learning_rate, BATCH_SIZE)
    epoch_batches = int(mnist.train.num_examples / BATCH_SIZE)

    for epoch in range(NB_EPOCHS):
        print('\nEpoch:', epoch)
        start = time.time()

        total_loss = 0
        for i in range(epoch_batches):
            batch_xs, _ = mnist.train.next_batch(BATCH_SIZE)
            total_loss += learner.train_network(batch_xs, None, LEARNING_RATE)

        train_accuracy = learner.test_network(train_loader)
        test_accuracy = learner.test_network(test_loader)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / epoch_batches)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)

    learner.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--beta', metavar='int', type=float, const=BETA, nargs='?', default=BETA,
        help='the value of beta, mutual information regulariser')
    parser.add_argument(
        '--rate', metavar='int', type=float, const=LEARNING_RATE, nargs='?', default=LEARNING_RATE,
        help='the learning rate for the Adam optimiser')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of rnn layers')

    args = parser.parse_args()
    main(args.beta, args.rate, args.layers)
