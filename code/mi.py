from tensorflow.examples.tutorials.mnist import input_data
from networks import StochasticFeedForwardNetwork
from learners import SupervisedLossLearner
import argparse
import time
from tools import Batcher

DATA_DIR = '/tmp/tensorflow/mnist/input_data'


def main(beta, learning_rate, nb_epochs, train_size, test_size,
         hidden_units, bottleneck_size, batch_size, nb_samples, update_marginal):
    run_name = 'sfnn_mnist_' + str(int(time.time()))
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    input_size = mnist.train.images.shape[1]
    output_size = mnist.train.labels.shape[1]

    sfnn = StochasticFeedForwardNetwork(input_size, hidden_units, bottleneck_size, output_size,
                                        update_marginal, nb_samples)
    learner = SupervisedLossLearner(sfnn, beta, learning_rate, batch_size, run_name)
    epoch_batches = int(mnist.train.num_examples / batch_size)

    train_data = mnist.train.images
    train_labels = mnist.train.labels
    test_data = mnist.test.images
    test_labels = mnist.test.labels
    if train_size:
        train_data = mnist.train.images[:train_size, :]
        train_labels = mnist.train.labels[:train_size]
    if test_size:
        test_data = mnist.test.images[:test_size, :]
        test_labels = mnist.test.labels[:test_size]

    train_loader = Batcher(train_data, train_labels, batch_size)
    test_loader = Batcher(test_data, test_labels, batch_size)
    best_accuracy = 0
    best_loss = None

    for epoch in range(nb_epochs):
        print('\nEpoch:', epoch)
        start = time.time()
        train_loader.reset_batch_pointer()

        total_loss = 0
        for i in range(epoch_batches):
            batch_xs, batch_ys = train_loader.next_batch()
            current_loss, loss_summary = learner.train_network(
                batch_xs, batch_ys, learning_rate)
            total_loss += current_loss

            learner.writer.add_summary(loss_summary, epoch * train_loader.num_batches + i)

        train_loss, train_accuracy = learner.test_network(train_loader, epoch=None)
        test_loss, test_accuracy = learner.test_network(test_loader, epoch)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / epoch_batches)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)
        print('Train loss: ', train_loss, ', test loss: ', test_loss)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print('-----')
            print('### Best accuracy ###')
            print('-----')
        if best_loss is None or test_loss < best_loss:
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
    parser.add_argument('--train', type=int,
                        help='train samples')
    parser.add_argument('--test', type=int,
                        help='test samples')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to run')
    parser.add_argument('--hidden', type=int, default=1024,
                        help='hidden units')
    parser.add_argument('--bottleneck', type=int, default=256,
                        help='bottleneck size')
    parser.add_argument('--batch', type=int, default=500,
                        help='batch size')
    parser.add_argument('--samples', type=int, default=1,
                        help='number of samples to get posterior expectation')
    parser.add_argument('--update_marginal', type=int, default=1,
                        help='marginal has learnable variable mean and variance')

    args = parser.parse_args()
    main(args.beta, args.rate, args.epochs,
         args.train, args.test, args.hidden, args.bottleneck, args.batch, args.samples,
         bool(args.update_marginal))
