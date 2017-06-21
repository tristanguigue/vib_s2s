import tensorflow as tf


def tf_binarize(images, threshold=0.1):
    return tf.cast(threshold < images, tf.float32)


def kl_divergence_with_std(mu, sigma):
    return 0.5 * tf.reduce_sum(
        tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)