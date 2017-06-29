from tensorflow.examples.tutorials.mnist import input_data
from networks import StochasticRNN
from learners import PredictionLossLearner
import argparse
import time

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
HIDDEN_SIZE = 128
BOTTLENECK_SIZE = 32
NB_EPOCHS = 1000
TRAIN_BATCH = 200
LEARNING_RATE = 0.001
BETA = 0.001
LEARNING_RATE_INCREASE_DELTA = 10


def cut_seq(seq, start_pos, seq_length):
    return seq[:, start_pos:start_pos + seq_length]


def main(beta, learning_rate, start_pos, seq_length, layers, examples):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    if not seq_length:
        seq_length = mnist.train.images.shape[1]
    if not examples:
        examples = mnist.train.num_examples

    srnn = StochasticRNN(seq_length, HIDDEN_SIZE, BOTTLENECK_SIZE, 1, layers)
    learner = PredictionLossLearner(srnn, beta, learning_rate, TRAIN_BATCH)
    epoch_batches = int(examples / TRAIN_BATCH)
    former_loss = None
    last_update = None

    for epoch in range(NB_EPOCHS):
        print('\nEpoch:', epoch)
        start = time.time()

        total_loss = 0
        for i in range(epoch_batches):
            batch_xs, _ = mnist.train.next_batch(TRAIN_BATCH)
            batch_xs = cut_seq(batch_xs, start_pos, seq_length)
            total_loss += learner.train_network(batch_xs, None, learning_rate)

        if former_loss is not None and total_loss >= former_loss:
            learning_rate /= 2
        elif last_update is not None and epoch - last_update > LEARNING_RATE_INCREASE_DELTA:
            learning_rate *= 2
            last_update = epoch
        former_loss = total_loss

        train_accuracy = learner.test_network(
            cut_seq(mnist.train.images[:examples], start_pos, seq_length), None)
        test_accuracy = learner.test_network(
            cut_seq(mnist.test.images, start_pos, seq_length), None)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / epoch_batches)
        print('Learning rate: ', learning_rate)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)

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
    parser.add_argument('--examples', type=int,
                        help='number of training examples')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of rnn layers')

    args = parser.parse_args()
    main(args.beta, args.rate, args.start, args.length, args.layers, args.examples)
