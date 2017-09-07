"""Apply the variational information bottleneck to predict the label of the first MNIST images in a
sequence.
"""
from tensorflow.examples.tutorials.mnist import input_data
from networks import Seq2Label
from learners import DiscreteLossLearner
from tools import Batcher
import argparse
import time
import os
import numpy as np

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
CHECKPOINT_PATH = 'checkpoints/'
DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
SAMPLE_EVERY = 100
NB_PRED_SAMPLES = 4


def main(beta, learning_rate, seq_length, layers, train_samples, test_samples,
         epochs, hidden_units, bottleneck_size, label_selected, batch_size, test_batch,
         save_checkpoints, nb_samples, update_marginal, dropout):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    run_name = 's2s_imsingledigit_' + str(int(time.time()))

    input_size = mnist.train.images.shape[1]
    output_size = mnist.train.labels.shape[1]

    train_data = mnist.train.images
    test_data = mnist.test.images
    train_labels = mnist.train.labels
    test_labels = mnist.test.labels

    if not train_samples:
        train_samples = int(train_data.shape[0] / seq_length)
    if not test_samples:
        test_samples = int(test_data.shape[0] / seq_length)
    train_data = train_data[:train_samples * seq_length, :]
    test_data = test_data[:test_samples * seq_length, :]
    train_labels = train_labels[:train_samples * seq_length, :]
    test_labels = test_labels[:test_samples * seq_length, :]

    train_data = np.array(np.split(train_data, train_samples))
    test_data = np.array(np.split(test_data, test_samples))
    train_labels = np.array(np.split(train_labels, train_samples))[:, 0, :]
    test_labels = np.array(np.split(test_labels, test_samples))[:, 0, :]

    train_loader = Batcher(train_data, train_labels, batch_size)
    test_loader = Batcher(test_data, test_labels, test_batch)
    seq2seq = Seq2Label(seq_length, hidden_units, bottleneck_size, input_size,
                        output_size, layers, nb_samples, update_marginal=True,
                        dropout=dropout)
    learner = DiscreteLossLearner(seq2seq, beta, learning_rate, batch_size, run_name)
    best_loss = None
    best_accuracy = 0

    for epoch in range(epochs):
        print('\nEpoch:', epoch)
        start = time.time()
        train_loader.reset_batch_pointer()

        total_loss = 0
        for i in range(train_loader.num_batches):
            batch_xs, batch_ys = train_loader.next_batch()
            current_loss, loss_summary = learner.train_network(batch_xs, batch_ys, learning_rate)
            total_loss += current_loss

            learner.writer.add_summary(loss_summary, epoch * train_loader.num_batches + i)

        learner.training_saver.save(learner.sess, DIR + CHECKPOINT_PATH + run_name)
        learner.saver.restore(learner.test_sess, DIR + CHECKPOINT_PATH + run_name)

        train_loss, train_accuracy = learner.test_network(train_loader, epoch=None)
        test_loss, test_accuracy = learner.test_network(test_loader, epoch)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / train_loader.num_batches)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)
        print('Train loss: ', train_loss, ', test loss: ', test_loss)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print('-----')
            print('### Best accuracy ###')
            print('-----')
        if best_loss is None or test_loss < best_loss:
            if save_checkpoints:
                learner.saver.save(learner.sess, DIR + CHECKPOINT_PATH + run_name)
            best_loss = test_loss
            print('-----')
            print('### Best loss ###')
            print('-----')

    learner.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=0.001,
                        help='the value of beta, mutual information regulariser')
    parser.add_argument('--rate', type=float, default=0.0001,
                        help='the learning rate for the Adam optimiser')
    parser.add_argument('--length', type=int,
                        help='length of sequence')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of rnn layers')
    parser.add_argument('--train', type=int,
                        help='train samples')
    parser.add_argument('--test', type=int,
                        help='test samples')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs to run')
    parser.add_argument('--hidden', type=int, default=128,
                        help='hidden units')
    parser.add_argument('--bottleneck', type=int, default=32,
                        help='bottleneck size')
    parser.add_argument('--label', type=int,
                        help='label of images selected')
    parser.add_argument('--batch', type=int, default=100,
                        help='batch size')
    parser.add_argument('--test_batch', type=int, default=500,
                        help='batch size')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='save checkpoints')
    parser.add_argument('--samples', type=int, default=1,
                        help='number of samples to get posterior expectation')
    parser.add_argument('--update_marginal', type=int, default=0,
                        help='marginal has learnable variable mean and variance')
    parser.add_argument('--dropout', type=int, default=0,
                        help='dropout regulariser')

    args = parser.parse_args()
    main(args.beta, args.rate, args.length, args.layers, args.train, args.test, args.epochs,
         args.hidden, args.bottleneck, args.label, args.batch, args.test_batch,
         bool(args.checkpoint), args.samples, bool(args.update_marginal), bool(args.dropout))