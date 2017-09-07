import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

DATA_DIR = '/tmp/tensorflow/mnist/input_data'
seq_size = 20
hidden_size = 64
output_size = 1
start_pos = 300
train_batch = 2
examples = 100
learning_rate = 0.005
layers = 2


def tf_binarize(images, threshold=0.1):
    return tf.cast(threshold < images, tf.float32)

mnist = input_data.read_data_sets(DATA_DIR)

stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(layers)])

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
    inputs = tf.expand_dims(tf_binarize(x), 2)
    lr = tf.placeholder(tf.float32)

with tf.name_scope('decoder'):
    decoder_weights = tf.get_variable('decoder_weights', shape=[hidden_size, output_size],
        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    decoder_biases = tf.Variable(tf.constant(0.0, shape=[output_size]), name='decoder_biases')

outputs, state = tf.nn.dynamic_rnn(stack, inputs, dtype=tf.float32)
flat_outputs = tf.reshape(outputs, [-1, hidden_size])

decoder_output = tf.matmul(flat_outputs, decoder_weights) + decoder_biases
decoder_output = tf.reshape(decoder_output, [-1, seq_size, output_size])

true_pixels = inputs[:, 1:]
pred_logits = tf.sigmoid(decoder_output[:, :-1])
predicted_pixels = tf.round(pred_logits)
accurate_predictions = tf.equal(predicted_pixels, true_pixels)
accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=true_pixels, logits=decoder_output[:, :-1])

loss_op = tf.reduce_mean(cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss_op)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
former_loss = None
last_update = None

train_data = mnist.train.images[mnist.train.labels == 1]
test_data = mnist.test.images[mnist.test.labels == 1]

for epoch in range(10000):
    print('\nEpoch:', epoch)
    start = time.time()

    batch_xs = train_data[:examples, start_pos:start_pos + seq_size]

    train_accuracy, train_loss = sess.run([accuracy, loss_op], feed_dict={x: batch_xs})
    test_accuracy, test_loss = sess.run([accuracy, loss_op], feed_dict={
        x: test_data[:1000, start_pos:start_pos + seq_size]
    })

    _, current_loss = sess.run([train_step, loss_op], feed_dict={
        x: batch_xs,
        lr: learning_rate
    })

    if former_loss is not None and current_loss >= former_loss:
        learning_rate /= 2
        last_update = epoch
    elif last_update is not None and epoch - last_update > 20:
        learning_rate *= 2
        last_update = epoch
    former_loss = current_loss

    print('Time: ', time.time() - start)
    print('Loss: ', current_loss)
    print('Learning rate: ', learning_rate)
    print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)
    print('Train loss: ', train_loss, ', test loss: ', test_loss)
