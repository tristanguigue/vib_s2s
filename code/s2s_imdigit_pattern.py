from tensorflow.examples.tutorials.mnist import input_data
from networks import Seq2Labels
from learners import SupervisedLossLearner
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
         epochs, hidden1_units, hidden2_units, bottleneck_size, label_selected, batch_size, test_batch,
         save_checkpoints, nb_samples, update_marginal, dropout):
    mnist = input_data.read_data_sets(DATA_DIR)
    mnist_onehot = input_data.read_data_sets(DATA_DIR, one_hot=True)
    run_name = 's2s_imdigit_' + str(int(time.time()))

    input_size = mnist.train.images.shape[1]
    output_size = mnist_onehot.train.labels.shape[1]

    train_data = mnist.train.images
    test_data = mnist.test.images

    if not train_samples:
        train_samples = int(train_data.shape[0] / seq_length)
    if not test_samples:
        test_samples = int(test_data.shape[0] / seq_length)
    train_data = train_data[:train_samples * seq_length, :]
    test_data = test_data[:test_samples * seq_length, :]
    train_labels = mnist.train.labels[:train_samples * seq_length]
    train_labels_onehot = mnist_onehot.train.labels[:train_samples * seq_length]
    test_labels = mnist.test.labels[:test_samples * seq_length]
    test_labels_onehot = mnist_onehot.test.labels[:test_samples * seq_length]

    train_data = np.reshape(train_data, [-1, seq_length, input_size])
    test_data = np.reshape(test_data, [-1, seq_length, input_size])
    train_labels = np.reshape(train_labels, [-1, seq_length])
    test_labels = np.reshape(test_labels, [-1, seq_length])
    train_labels_onehot = np.reshape(train_labels_onehot, [-1, seq_length, output_size])
    test_labels_onehot = np.reshape(test_labels_onehot, [-1, seq_length, output_size])

    ids = np.expand_dims(np.arange(test_samples), 1)
    indices = np.argsort(test_labels, axis=1)
    print(test_labels_onehot.shape)
    test_labels = test_labels_onehot[ids, indices, :]
    test_data = test_data[ids, indices, :]

    ids = np.expand_dims(np.arange(train_samples), 1)
    indices = np.argsort(train_labels, axis=1)
    train_labels = train_labels_onehot[ids, indices, :]
    train_data = train_data[ids, indices, :]

    train_loader = Batcher(train_data, train_labels, batch_size)
    test_loader = Batcher(test_data, test_labels, test_batch)
    seq2seq = Seq2Labels(seq_length, hidden1_units, hidden2_units, bottleneck_size, input_size,
                         output_size, layers, nb_samples, update_prior=update_marginal,
                         dropout=dropout)
    learner = SupervisedLossLearner(seq2seq, beta, learning_rate, batch_size, run_name,
                                    reduce_seq=True)
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

        if SAMPLE_EVERY is not None and not epoch % SAMPLE_EVERY:
            train_samples = learner.predict_sequence(
                train_data[:NB_PRED_SAMPLES], train_labels[:NB_PRED_SAMPLES])
            test_samples = learner.predict_sequence(
                test_data[:NB_PRED_SAMPLES], test_labels[:NB_PRED_SAMPLES])
            print(train_samples)
            print(test_samples)

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
    parser.add_argument('--hidden1', type=int, default=128,
                        help='hidden units of encoder')
    parser.add_argument('--hidden2', type=int, default=16,
                        help='hidden units of decoder')
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
    parser.add_argument('--samples', type=int, default=12,
                        help='number of samples to get posterior expectation')
    parser.add_argument('--update_marginal', type=int, default=0,
                        help='marginal has learnable variable mean and variance')
    parser.add_argument('--dropout', type=int, default=0,
                        help='dropout regulariser')

    args = parser.parse_args()
    main(args.beta, args.rate, args.length, args.layers, args.train, args.test, args.epochs,
         args.hidden1, args.hidden2, args.bottleneck, args.label, args.batch, args.test_batch,
         bool(args.checkpoint), args.samples, bool(args.update_marginal), bool(args.dropout))
