import os
import numpy as np
from networks import StochasticRNN
from learners import PredictionLossLearner

DATA_DIR = 'data/binary_samples10000_s60.npy'
HIDDEN_SIZE = 128
BOTTLENECK_SIZE = 32
PREDICT_SAMPLES = 5
DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
RUN_NAME = 'srnn_generated_1500028572.738754'
CHECKPOINT_PATH = 'checkpoints/'
SEQ_LENGTH = 60


def main():
    srnn = StochasticRNN(SEQ_LENGTH, HIDDEN_SIZE, BOTTLENECK_SIZE, 1, 1, True, True)
    learner = PredictionLossLearner(srnn, None, None, None, RUN_NAME)
    learner.saver.restore(learner.sess, DIR + CHECKPOINT_PATH + RUN_NAME)

    data = np.load(DIR + DATA_DIR)
    train_data = data[PREDICT_SAMPLES:]
    test_data = data[-PREDICT_SAMPLES:]
    predicted_train_sequences = learner.predict_sequence(train_data)
    predicted_test_sequences = learner.predict_sequence(test_data)

    print('Train data')
    print(train_data[:, 1:])
    print('Predicted')
    print(predicted_train_sequences)

    print('Test data')
    print(test_data[:, 1:])
    print('Predicted')
    print(predicted_test_sequences)

if __name__ == '__main__':
    main()