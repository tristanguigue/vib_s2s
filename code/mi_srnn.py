from tensorflow.examples.tutorials.mnist import input_data
from networks import StochasticRNN
from learners import PredictionLossLearner
from tools import Batcher
import argparse
import time

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
HIDDEN_SIZE = 128
BOTTLENECK_SIZE = 32
NB_EPOCHS = 1000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
BETA = 0.001
LEARNING_RATE_INCREASE_DELTA = 10


def cut_seq(seq, start_pos, seq_length):
    return seq[:, start_pos:start_pos + seq_length]


def main(beta, learning_rate, start_pos, seq_length, layers):
    mnist = input_data.read_data_sets(DATA_DIR)
    if not seq_length:
        seq_length = mnist.train.images.shape[1]

    train_data = cut_seq(mnist.train.images[mnist.train.labels == 6], start_pos, seq_length)
    test_data = cut_seq(mnist.test.images[mnist.test.labels == 6], start_pos, seq_length)
    train_loader = Batcher(train_data, None, BATCH_SIZE)
    test_loader = Batcher(test_data, None, BATCH_SIZE)

    srnn = StochasticRNN(seq_length, HIDDEN_SIZE, BOTTLENECK_SIZE, 1, layers, True)
    learner = PredictionLossLearner(srnn, beta, learning_rate, BATCH_SIZE)
    former_loss = None
    last_update = 0

    for epoch in range(NB_EPOCHS):
        print('\nEpoch:', epoch)
        start = time.time()
        train_loader.reset_batch_pointer()

        total_loss = 0
        for i in range(train_loader.num_batches):
            batch_xs, _ = train_loader.next_batch()
            total_loss += learner.train_network(batch_xs, None, learning_rate)

        if former_loss is not None and total_loss >= former_loss:
            learning_rate /= 2
        elif epoch - last_update > LEARNING_RATE_INCREASE_DELTA:
            learning_rate *= 2
            last_update = epoch
        former_loss = total_loss

        train_loss, train_accuracy = learner.test_network(train_loader)
        test_loss, test_accuracy = learner.test_network(test_loader)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / train_loader.num_batches)
        print('Learning rate: ', learning_rate)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)
        print('Train loss: ', train_loss, ', test loss: ', test_loss)

    learner.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=BETA,
                        help='the value of beta, mutual information regulariser')
    parser.add_argument('--rate', type=float, default=LEARNING_RATE,
                        help='the learning rate for the Adam optimiser')
    parser.add_argument('--start', type=int, default=0,
                        help='start position in sequence')
    parser.add_argument('--length', type=int,
                        help='length of sequence')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of rnn layers')

    args = parser.parse_args()
    main(args.beta, args.rate, args.start, args.length, args.layers)
