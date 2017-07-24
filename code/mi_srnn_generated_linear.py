from networks import StochasticRNN
from learners import LinearPredictionLossLearner
from tools import Batcher
import argparse
import time
import numpy as np
import os

DATA_DIR = 'data/linear_samples.npy'
CHECKPOINT_PATH = 'checkpoints/'
DIR = os.path.dirname(os.path.realpath(__file__)) + '/'


def cut_seq(seq, start_pos, seq_length):
    return seq[:, start_pos:start_pos + seq_length]


def main(beta, learning_rate, start_pos, seq_length, layers, nb_epochs, train_size, test_size,
         hidden_units, bottleneck_size, batch_size, lstm_cell):
    data = np.load(DIR + DATA_DIR)
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    run_name = 'srnn_gen_linear_' + str(int(time.time()))

    if not seq_length:
        seq_length = train_data.shape[1]

    train_data = cut_seq(train_data, start_pos, seq_length)
    test_data = cut_seq(test_data, start_pos, seq_length)
    train_loader = Batcher(train_data, None, batch_size)
    test_loader = Batcher(test_data, None, batch_size)
    best_loss = None
    best_train_loss = None

    srnn = StochasticRNN(seq_length, hidden_units, bottleneck_size, 1, layers, True, lstm_cell, False)
    learner = LinearPredictionLossLearner(srnn, beta, learning_rate, batch_size, run_name)

    for epoch in range(nb_epochs):
        print('\nEpoch:', epoch)
        start = time.time()
        train_loader.reset_batch_pointer()

        total_loss = 0
        for i in range(train_loader.num_batches):
            batch_xs, _ = train_loader.next_batch()
            current_loss, lr_summary, loss_summary = learner.train_network(
                batch_xs, None, learning_rate)
            total_loss += current_loss

            learner.writer.add_summary(lr_summary, epoch * train_loader.num_batches + i)
            learner.writer.add_summary(loss_summary, epoch * train_loader.num_batches + i)

        train_loss, _ = learner.test_network(train_loader, epoch=None)
        test_loss, _ = learner.test_network(test_loader, epoch)

        if best_loss is None or test_loss < best_loss:
            learner.saver.save(learner.sess, DIR + CHECKPOINT_PATH + run_name)
            best_loss = test_loss

        if best_train_loss is None or train_loss < best_train_loss:
            learner.saver.save(learner.sess, DIR + CHECKPOINT_PATH + 'train_' + run_name)
            best_train_loss = train_loss

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / train_loader.num_batches)
        print('Learning rate: ', learning_rate)
        print('Train loss: ', train_loss, ', test loss: ', test_loss)

    learner.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=0.001,
                        help='the value of beta, mutual information regulariser')
    parser.add_argument('--rate', type=float, default=0.0005,
                        help='the learning rate for the Adam optimiser')
    parser.add_argument('--start', type=int, default=0,
                        help='start position in sequence')
    parser.add_argument('--length', type=int,
                        help='length of sequence')
    parser.add_argument('--train', type=int, default=500,
                        help='train samples')
    parser.add_argument('--test', type=int, default=500,
                        help='test samples')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of rnn layers')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs to run')
    parser.add_argument('--hidden', type=int, default=128,
                        help='hidden units')
    parser.add_argument('--bottleneck', type=int, default=32,
                        help='bottleneck size')
    parser.add_argument('--batch', type=int, default=500,
                        help='batch size')
    parser.add_argument('--lstm', type=int, default=1,
                        help='is lstm cell')

    args = parser.parse_args()
    main(args.beta, args.rate, args.start, args.length, args.layers, args.epochs,
         args.train, args.test, args.hidden, args.bottleneck, args.batch, bool(args.lstm))
