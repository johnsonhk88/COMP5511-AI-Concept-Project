"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 100
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.001               # learning rate
FORCE_REGEN_MDL = False

# data
mnist = input_data.read_data_sets('./mnist', one_hot=True, validation_size=0)              # they has been normalized to range (0,1)
test_x = mnist.test.images
test_y = mnist.test.labels
test_y_am = np.argmax(test_y, 1)

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
print(test_x.shape)
print(test_y.shape)
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0]))
# plt.show()

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # shape(batch, 784)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])                             # input y

# RNN
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], 10)              # output based on the last output step

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
import sys
import os
saver = tf.train.Saver()
cp_path = os.path.join("model", os.path.basename(__file__), "checkpoint")
model_path = os.path.join("model", os.path.basename(__file__), os.path.basename(sys.argv[0]) + ".ckpt")
if not os.path.exists(cp_path) or FORCE_REGEN_MDL:
    sess.run(init_op)     # initialize var in graph

    for step in range(600):    # training
        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
        if step % 50 == 0 or step == 599:      # testing
            print('Step:', step, '| train loss: %.4f' % loss_)
            # accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
            # print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
    save_path = saver.save(sess, model_path)
    print("Model saved in path: %s" % save_path)
else:
    # Restore variables from disk.
    saver.restore(sess, model_path)
    print("Model restored.")

test_output = sess.run(output, {tf_x: test_x})
pred_y = np.argmax(test_output, 1)
print(test_y.shape)
print(test_output.shape)

from sklearn import metrics
print("Classification report:\n%s\n" % (metrics.classification_report(test_y_am, pred_y)))
print("Confusion matrix:\n%s\n" % (metrics.confusion_matrix(test_y_am, pred_y)))

# print 10 predictions from test data
# test_output = sess.run(output, {tf_x: test_x[:10]})
# pred_y = np.argmax(test_output, 1)
# print(pred_y, 'prediction number')
# print(np.argmax(test_y[:10], 1), 'real number')