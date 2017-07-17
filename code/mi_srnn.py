from tensorflow.examples.tutorials.mnist import input_data
from networks import StochasticRNN
from learners import PredictionLossLearner
from tools import Batcher
import argparse
import time
import os

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
HIDDEN_SIZE = 128
BOTTLENECK_SIZE = 32
NB_EPOCHS = 5000
BATCH_SIZE = 500
LEARNING_RATE = 0.0005
BETA = 0.001
TRAIN_SIZE = 500
TEST_SIZE = 500
CHECKPOINT_PATH = 'checkpoints/'
DIR = os.path.dirname(os.path.realpath(__file__)) + '/'


def main(beta, learning_rate, start_pos, seq_length, layers, train_samples, test_samples, epochs,
         hidden_units, bottleneck_size, label_selected):
    mnist = input_data.read_data_sets(DATA_DIR)
    if not seq_length:
        seq_length = mnist.train.images.shape[1]
    run_name = 'srnn_mnist_' + str(int(time.time()))

    train_data = mnist.train.images
    test_data = mnist.test.images
    if label_selected:
        train_data = mnist.train.images[mnist.train.labels == label_selected]
        test_data = mnist.test.images[mnist.test.labels == label_selected]
    train_data = train_data[:train_samples, start_pos:start_pos + seq_length]
    test_data = test_data[:test_samples, start_pos:start_pos + seq_length]

    train_loader = Batcher(train_data, None, BATCH_SIZE)
    test_loader = Batcher(test_data, None, BATCH_SIZE)

    srnn = StochasticRNN(seq_length, hidden_units, bottleneck_size, 1, layers, True, True)
    learner = PredictionLossLearner(srnn, beta, learning_rate, BATCH_SIZE, run_name)
    best_loss = None

    for epoch in range(epochs):
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

        train_loss, train_accuracy = learner.test_network(train_loader, epoch=None)
        test_loss, test_accuracy = learner.test_network(test_loader, epoch)

        if best_loss is None or test_loss < best_loss:
            learner.saver.save(learner.sess, DIR + CHECKPOINT_PATH + run_name)
            best_loss = test_loss

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
    parser.add_argument('--train', type=int, default=TRAIN_SIZE,
                        help='train samples')
    parser.add_argument('--test', type=int, default=TEST_SIZE,
                        help='test samples')
    parser.add_argument('--epochs', type=int, default=NB_EPOCHS,
                        help='number of epochs to run')
    parser.add_argument('--hidden', type=int, default=HIDDEN_SIZE,
                        help='hidden units')
    parser.add_argument('--bottleneck', type=int, default=BOTTLENECK_SIZE,
                        help='bottleneck size')
    parser.add_argument('--label', type=int,
                        help='label of images selected')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                        help='batch size')


    args = parser.parse_args()
    main(args.beta, args.rate, args.start, args.length, args.layers, args.train, args.test,
         args.epochs, args.hidden, args.bottleneck, args.label, args.batch)
