"""Apply the variational information bottleneck to predict the N next pixels in an MNIST image."""
from tensorflow.examples.tutorials.mnist import input_data
from networks import Seq2Seq
from learners import DiscreteLossLearner
from tools import Batcher
import argparse
import time
import os

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
CHECKPOINT_PATH = 'checkpoints/'
DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
SAMPLE_EVERY = 100
NB_SAMPLES = 4


def main(beta, learning_rate, start_pos, partial_seq_length, layers, train_samples, test_samples,
         epochs, hidden1_units, hidden2_units, bottleneck_size, label_selected, batch_size, test_batch,
         output_seq_size, save_checkpoints, nb_samples, update_marginal, dropout):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    if not partial_seq_length:
        partial_seq_length = mnist.train.images.shape[1]
    run_name = 's2s_mnist_' + str(int(time.time()))

    train_data = mnist.train.images
    test_data = mnist.test.images
    if label_selected:
        train_data = mnist.train.images[mnist.train.labels == label_selected]
        test_data = mnist.test.images[mnist.test.labels == label_selected]
    if train_samples:
        train_data = train_data[:train_samples, start_pos:start_pos + partial_seq_length + output_seq_size]
    else:
        train_data = train_data[:, start_pos:start_pos + partial_seq_length + output_seq_size]
    if test_samples:
        test_data = test_data[:test_samples, start_pos:start_pos + partial_seq_length + output_seq_size]
    else:
        test_data = test_data[:, start_pos:start_pos + partial_seq_length + output_seq_size]

    train_loader = Batcher(train_data, None, batch_size)
    test_loader = Batcher(test_data, None, test_batch)
    seq2seq = Seq2Seq(partial_seq_length, output_seq_size, hidden1_units, hidden2_units,
                      bottleneck_size, 1, layers, nb_samples, update_marginal=update_marginal, dropout=dropout)
    learner = DiscreteLossLearner(seq2seq, beta, learning_rate, batch_size, run_name, binary=True,
                                  reduce_seq=True)
    best_loss = None
    best_accuracy = 0

    for epoch in range(epochs):
        print('\nEpoch:', epoch)
        start = time.time()
        train_loader.reset_batch_pointer()

        total_loss = 0
        for i in range(train_loader.num_batches):
            batch_xs, _ = train_loader.next_batch()
            current_loss, loss_summary = learner.train_network(
                batch_xs, None, learning_rate)
            total_loss += current_loss

            learner.writer.add_summary(loss_summary, epoch * train_loader.num_batches + i)

        learner.training_saver.save(learner.sess, DIR + CHECKPOINT_PATH + run_name)
        learner.saver.restore(learner.test_sess, DIR + CHECKPOINT_PATH + run_name)

        train_loss, train_accuracy = learner.test_network(train_loader, epoch=None)
        test_loss, test_accuracy = learner.test_network(test_loader, epoch)

        if SAMPLE_EVERY is not None and not epoch % SAMPLE_EVERY:
            train_samples = learner.sample_sequence(train_data[:NB_SAMPLES])
            test_samples = learner.sample_sequence(test_data[:NB_SAMPLES])
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
    parser.add_argument('--start', type=int, default=0,
                        help='start position in sequence')
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
    parser.add_argument('--output_seq_size', type=int, default=15,
                        help='output sequence size')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='save checkpoints')
    parser.add_argument('--samples', type=int, default=1,
                        help='number of samples to get posterior expectation')
    parser.add_argument('--update_marginal', type=int, default=0,
                        help='marginal has learnable variable mean and variance')
    parser.add_argument('--dropout', type=int, default=0,
                        help='dropout regulariser')

    args = parser.parse_args()
    main(args.beta, args.rate, args.start, args.length, args.layers, args.train, args.test, args.epochs,
         args.hidden1, args.hidden2, args.bottleneck, args.label, args.batch, args.test_batch, args.output_seq_size,
         bool(args.checkpoint), args.samples, bool(args.update_marginal), bool(args.dropout))
