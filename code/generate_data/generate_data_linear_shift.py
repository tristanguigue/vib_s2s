import numpy as np
import matplotlib.pyplot as plt

seq_size = 30


train_samples = []
start_x, start_y = np.random.multivariate_normal([-10, -8], [[1, 0], [0, 1]], 1000).T

for i in range(1000):
    sequence = []
    for t in range(seq_size):
        sequence.append(start_y[i] + ((t - 10) / 10)**3 + np.random.normal(scale=0.3))
    train_samples.append(sequence)

start_x, start_y = np.random.multivariate_normal([7, 3], [[1, 0], [0, 1]], 1000).T
test_samples = []
for i in range(1000):
    sequence = []
    for t in range(seq_size):
        sequence.append(start_y[i] + ((t - 10) / 10)**3 + np.random.normal(scale=0.3))
    test_samples.append(sequence)

train_samples = np.asarray(train_samples)
test_samples = np.asarray(test_samples)

min_val = min(np.min(train_samples), np.min(test_samples))
max_val = max(np.max(train_samples), np.max(test_samples))
train_samples = (train_samples - min_val) / (max_val - min_val)
test_samples = (test_samples - min_val) / (max_val - min_val)


plt.plot(train_samples[0, :])
plt.plot(train_samples[1, :])
plt.plot(train_samples[2, :])
plt.plot(train_samples[3, :])
plt.plot(train_samples[4, :])
plt.plot(train_samples[5, :])
plt.plot(test_samples[0, :])
plt.plot(test_samples[1, :])
plt.plot(test_samples[2, :])
plt.plot(test_samples[3, :])
plt.show()


np.save('linear_shift_train_samples.npy', train_samples)
np.save('linear_shift_test_samples.npy', test_samples)
