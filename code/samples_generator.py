import numpy as np


def sample_bernouilli(input_values):
    return np.random.binomial(1, p=input_values)


SEQ_SIZE = 30

samples = []
for i in range(10000):
    ps = [0.5] * 5
    sequence = sample_bernouilli(ps)
    for t in range(5, SEQ_SIZE):
        si = np.sign(sum(sequence[t - 5:t]) - 5 / 2)
        if si == -1:
            p = 0.9
        else:
            p = 0.1
        s = sample_bernouilli(p)
        sequence = np.append(sequence, s)
    samples.append(sequence)
samples = np.asarray(samples)

print(samples)
np.save('data/binary_samples10000.npy', samples)
