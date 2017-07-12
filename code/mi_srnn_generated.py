from tensorflow.examples.tutorials.mnist import input_data
from networks import StochasticRNN
from learners import PredictionLossLearner
from tools import Batcher
import argparse
import time
import numpy as np

DATA_DIR = 'data/generated_samples.npy'
HIDDEN_SIZE = 128
BOTTLENECK_SIZE = 128
NB_EPOCHS = 2000
BATCH_SIZE = 500
LEARNING_RATE = 0.0005
BETA = 0.001
TRAIN_TEST_SPLIT = 500


def cut_seq(seq, start_pos, seq_length):
    return seq[:, start_pos:start_pos + seq_length]


def main(beta, learning_rate, start_pos, seq_length, layers, nb_epochs):
    data = np.load(DATA_DIR)
    train_data = data[:TRAIN_TEST_SPLIT]
    test_data = data[TRAIN_TEST_SPLIT:]

    if not seq_length:
        seq_length = train_data.shape[1]

    train_data = cut_seq(train_data, start_pos, seq_length)
    test_data = cut_seq(test_data, start_pos, seq_length)
    train_loader = Batcher(train_data, None, BATCH_SIZE)
    test_loader = Batcher(test_data, None, BATCH_SIZE)

    srnn = StochasticRNN(seq_length, HIDDEN_SIZE, BOTTLENECK_SIZE, 1, layers, True, False)
    learner = PredictionLossLearner(srnn, beta, learning_rate, BATCH_SIZE)

    for epoch in range(nb_epochs):
        print('\nEpoch:', epoch)
        start = time.time()
        train_loader.reset_batch_pointer()

        total_loss = 0
        for i in range(train_loader.num_batches):
            batch_xs, _ = train_loader.next_batch()
            total_loss += learner.train_network(batch_xs, None, learning_rate)

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
    parser.add_argument('--epochs', type=int, default=NB_EPOCHS,
                        help='number of epochs to run')

    args = parser.parse_args()
    main(args.beta, args.rate, args.start, args.length, args.layers, args.epochs)
