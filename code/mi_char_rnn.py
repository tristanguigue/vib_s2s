from networks import StochasticCharRNN
from learners import CharPredictionLossLearner
import argparse
import time
from tools import TextLoader

DATA_DIR = 'data/shakespeare'
HIDDEN_SIZE = 128
BOTTLENECK_SIZE = 32
NB_EPOCHS = 1000
TRAIN_BATCH = 200
SEQ_LENGTH = 20
LEARNING_RATE = 0.001
BETA = 0.001
LEARNING_RATE_INCREASE_DELTA = 10


def main(beta, learning_rate, train):
    train_data_loader = TextLoader(DATA_DIR, TRAIN_BATCH, SEQ_LENGTH, 'train_input.txt')
    test_data_loader = TextLoader(DATA_DIR, TRAIN_BATCH, SEQ_LENGTH, 'test_input.txt')

    vocab_size = train_data_loader.vocab_size

    srnn = StochasticCharRNN(SEQ_LENGTH, HIDDEN_SIZE, BOTTLENECK_SIZE, vocab_size, 1)
    learner = CharPredictionLossLearner(srnn, beta, learning_rate, TRAIN_BATCH)
    former_loss = None
    last_update = 0

    for epoch in range(NB_EPOCHS):
        print('\nEpoch:', epoch)
        train_data_loader.reset_batch_pointer()
        start = time.time()

        total_loss = 0
        for i in range(train_data_loader.num_batches):
            batch_xs, _ = train_data_loader.next_batch()
            total_loss += learner.train_network(batch_xs, None, learning_rate)

        train_accuracy = learner.test_network_loader(train_data_loader)
        test_accuracy = learner.test_network_loader(test_data_loader)

        if former_loss is not None and total_loss >= former_loss:
            learning_rate /= 2
            last_update = epoch
        elif epoch - last_update > LEARNING_RATE_INCREASE_DELTA:
            learning_rate *= 2
            last_update = epoch
        former_loss = total_loss

        print('Time: ', time.time() - start)
        print('Loss: ', total_loss / train_data_loader.num_batches)
        print('Learning rate: ', learning_rate)
        print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)

    learner.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=BETA,
        help='the value of beta, mutual information regulariser')
    parser.add_argument('--rate', type=float, default=LEARNING_RATE,
        help='the learning rate for the Adam optimiser')

    args = parser.parse_args()
    main(args.beta, args.rate, train=True)
