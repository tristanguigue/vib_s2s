from tensorflow.examples.tutorials.mnist import input_data
from networks import Seq2Seq
from learners import PartialPredictionLossLearner
from tools import Batcher
import argparse
import time
import os

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
CHECKPOINT_PATH = 'checkpoints/'
DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
SAMPLE_EVERY = 200
NB_SAMPLES = 4


def main(beta, learning_rate, layers, train_samples, test_samples, epochs,
         hidden_units, bottleneck_size, label_selected, batch_size, lstm_cell, output_seq_size,
         save_checkpoints):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    seq_size = mnist.train.images.shape[1]
    partial_sequence_size = int(1 * seq_size / 3)
    run_name = 's2s_mnist_' + str(int(time.time()))

    train_data = mnist.train.images
    test_data = mnist.test.images
    if label_selected:
        train_data = mnist.train.images[mnist.train.labels == label_selected]
        test_data = mnist.test.images[mnist.test.labels == label_selected]
    train_data = train_data[:train_samples, :]
    test_data = test_data[:test_samples, :]

    train_loader = Batcher(train_data, None, batch_size)
    test_loader = Batcher(test_data, None, batch_size)
    seq2seq = Seq2Seq(seq_size, partial_sequence_size, output_seq_size, hidden_units,
                      bottleneck_size, 1, layers, update_prior=True, lstm=lstm_cell)
    learner = PartialPredictionLossLearner(seq2seq, beta, learning_rate, batch_size, run_name)
    best_loss = None

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

        train_loss, train_accuracy = learner.test_network(train_loader, epoch=None)
        test_loss, test_accuracy = learner.test_network(test_loader, epoch)

        if save_checkpoints:
            if best_loss is None or test_loss < best_loss:
                learner.saver.save(learner.sess, DIR + CHECKPOINT_PATH + run_name)
                best_loss = test_loss

        if SAMPLE_EVERY is not None and not epoch % SAMPLE_EVERY:
            train_samples = learner.sample_sequence(train_data[:NB_SAMPLES])
            test_samples = learner.sample_sequence(test_data[:NB_SAMPLES])
            print(train_samples)
            print(test_samples)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / train_loader.num_batches)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)
        print('Train loss: ', train_loss, ', test loss: ', test_loss)

    learner.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=0.001,
                        help='the value of beta, mutual information regulariser')
    parser.add_argument('--rate', type=float, default=0.0005,
                        help='the learning rate for the Adam optimiser')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of rnn layers')
    parser.add_argument('--train', type=int, default=500,
                        help='train samples')
    parser.add_argument('--test', type=int, default=500,
                        help='test samples')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs to run')
    parser.add_argument('--hidden', type=int, default=128,
                        help='hidden units')
    parser.add_argument('--bottleneck', type=int, default=64,
                        help='bottleneck size')
    parser.add_argument('--label', type=int,
                        help='label of images selected')
    parser.add_argument('--batch', type=int, default=500,
                        help='batch size')
    parser.add_argument('--lstm', type=int, default=1,
                        help='is lstm cell')
    parser.add_argument('--output_seq_size', type=int, default=15,
                        help='output sequence size')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='save checkpoints')

    args = parser.parse_args()
    main(args.beta, args.rate, args.layers, args.train, args.test, args.epochs,
         args.hidden, args.bottleneck, args.label, args.batch, bool(args.lstm), args.output_seq_size,
         bool(args.checkpoint))
