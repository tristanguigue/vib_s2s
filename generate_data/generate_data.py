import tensorflow as tf


def sample_bernouilli(input_values, batch_size):
    return tf.where(
        tf.random_uniform([batch_size]) - input_values < 0,
        tf.ones([batch_size]),
        tf.zeros([batch_size]))

data = sample_bernouilli([0.8] * 100, 100)

sess = tf.InteractiveSession()
with sess.as_default():
    print(data.eval())