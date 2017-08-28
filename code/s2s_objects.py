from networks import Seq2Label
from learners import SupervisedLossLearner
from tools import Batcher
import argparse
import time
import os
from scipy import misc
import numpy as np
import glob

DATA_DIR = '/data/particle_box/'
CHECKPOINT_PATH = 'checkpoints/'
DIR = os.path.dirname(os.path.realpath(__file__)) + '/'


def main(beta, learning_rate, layers, train_samples, test_samples,
         epochs, hidden_units, bottleneck_size, batch_size,
         save_checkpoints, nb_samples, update_marginal):

    targets = np.load(DIR + DATA_DIR + 'targets.npy')
    targets = (targets - np.min(targets)) / (np.max(targets) - np.min(targets))

    videos = []
    for i in range(train_samples + test_samples):
        video = []
        for image_path in glob.glob(DIR + DATA_DIR + 'particle_box_' + str(i) + '/*.png'):
            rgb = misc.imread(image_path)
            image = np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])
            image = (128 > image).astype('float32')
            video.append(image)
        video = np.array(video)
        videos.append(video)
    videos = np.array(videos)

    seq_length = videos.shape[1]
    size_x = videos.shape[2]
    size_y = videos.shape[3]
    input_size = size_x * size_y
    videos = np.reshape(videos, [-1, seq_length, input_size])
    output_size = targets.shape[1]
    run_name = 's2s_objects_' + str(int(time.time()))

    train_data = videos[:train_samples, :]
    train_labels = targets[:train_samples]
    test_data = videos[train_samples:train_samples + test_samples, :]
    test_labels = targets[train_samples:train_samples + test_samples]

    train_loader = Batcher(train_data, train_labels, batch_size)
    test_loader = Batcher(test_data, test_labels, batch_size)
    seq2seq = Seq2Label(seq_length, hidden_units, bottleneck_size, input_size,
                        output_size, layers, nb_samples, update_prior=update_marginal)
    learner = SupervisedLossLearner(seq2seq, beta, learning_rate, batch_size, run_name, continuous=True)
    best_loss = None

    for epoch in range(epochs):
        print('\nEpoch:', epoch)
        start = time.time()
        train_loader.reset_batch_pointer()

        total_loss = 0
        for i in range(train_loader.num_batches):
            batch_xs, batch_ys = train_loader.next_batch()
            current_loss, loss_summary = learner.train_network(
                batch_xs, batch_ys, learning_rate)
            total_loss += current_loss

            learner.writer.add_summary(loss_summary, epoch * train_loader.num_batches + i)

        train_loss, _ = learner.test_network(train_loader, epoch=None)
        test_loss, _ = learner.test_network(test_loader, epoch)

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / train_loader.num_batches)
        print('Train loss: ', train_loss, ', test loss: ', test_loss)
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
    parser.add_argument('--layers', type=int, default=1,
                        help='number of rnn layers')
    parser.add_argument('--train', type=int, default=600,
                        help='train samples')
    parser.add_argument('--test', type=int, default=400,
                        help='test samples')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs to run')
    parser.add_argument('--hidden', type=int, default=128,
                        help='hidden units of decoder')
    parser.add_argument('--bottleneck', type=int, default=32,
                        help='bottleneck size')
    parser.add_argument('--batch', type=int, default=200,
                        help='batch size')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='save checkpoints')
    parser.add_argument('--samples', type=int, default=12,
                        help='number of samples to get posterior expectation')
    parser.add_argument('--update_marginal', type=int, default=0,
                        help='marginal has learnable variable mean and variance')

    args = parser.parse_args()
    main(args.beta, args.rate, args.layers, args.train, args.test, args.epochs,
         args.hidden, args.bottleneck, args.batch,
         bool(args.checkpoint), args.samples, bool(args.update_marginal))
