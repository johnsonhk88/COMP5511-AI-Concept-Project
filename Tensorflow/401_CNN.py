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
import sys
#import tensorflow.compat.v1 as tf

#tf.disable_v2_behavior()

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)

tf.set_random_seed(1)
np.random.seed(1)

NO_PLOT = True
BATCH_SIZE = 100
LR = 0.001  # learning rate
FORCE_REGEN_MDL = False

mnist = input_data.read_data_sets('./mnist', one_hot=True, validation_size=0)  # they has been normalized to range (0,1)
# mnist = tf.keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# test_x = mnist.test.images[:10000]
# test_y = mnist.test.labels[:10000]
test_x = mnist.test.images  # [:10000]
test_y = mnist.test.labels  # [:10000]
# test_x = X_test[:2000]
# test_y = y_test[:2000]
test_y_am = np.argmax(test_y, 1)

# plot one example
print(mnist.train.images.shape)  # (55000, 28 * 28)
print(mnist.train.labels.shape)  # (55000, 10)
# print(X_train.shape)
# print(y_train.shape)
print(test_x.shape)
print(test_y.shape)
# if not NO_PLOT:
#     plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
#     plt.title('%i' % np.argmax(mnist.train.labels[0]));
#     plt.show()

tf_x = tf.compat.v1.placeholder(tf.float32, [None, 28 * 28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])  # (batch, height, width, channel)
tf_y = tf.compat.v1.placeholder(tf.int32, [None, 10])  # input y

# CNN
conv1 = tf.layers.conv2d(  # shape (28, 28, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)  # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)  # -> (14, 14, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)  # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (7, 7, 32)
flat = tf.reshape(pool2, [-1, 7 * 7 * 32])  # -> (7*7*32, )
output = tf.layers.dense(flat, 10)  # output layer

loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)  # compute cost
train_op = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.compat.v1.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1), )[1]

sess = tf.compat.v1.Session()
init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.local_variables_initializer())  # the local var is for accuracy_op

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm

try:
    from sklearn.manifold import TSNE;
    HAS_SK = True
except:
    HAS_SK = False;
    print('\nPlease install sklearn for layer visualization\n')


def plot_with_labels(lowDWeights, labels):
    plt.cla();
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9));
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer');
    plt.show();
    plt.pause(0.01)


if not NO_PLOT:
    plt.ion()

import sys
import os
saver = tf.train.Saver()
cp_path = os.path.join("model", os.path.basename(__file__), "checkpoint")
model_path = os.path.join("model", os.path.basename(__file__), os.path.basename(sys.argv[0]) + ".ckpt")
if not os.path.exists(cp_path) or FORCE_REGEN_MDL:
    sess.run(init_op)  # initialize var in graph

    # def next_batch(num, data, labels):
    #     '''
    #     Return a total of `num` random samples and labels.
    #     '''
    #     idx = np.arange(0, len(data))
    #     np.random.shuffle(idx)
    #     idx = idx[:num]
    #     data_shuffle = [data[i] for i in idx]
    #     labels_shuffle = [labels[i] for i in idx]
    #
    #     return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    # X_train = X_train / 255
    # tmp_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # tmp_ds = tmp_ds.batch(1)
    # tmp_ds = tmp_ds.batch(BATCH_SIZE)

    for step in range(600):
        # iterator = tf.compat.v1.data.make_one_shot_iterator(tmp_ds)
        # for (b_x,b_y) in iterator.get_next():
        #     _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
        #     if step % 50 == 0:
        #         print('Step:', step, '| train loss: %.4f' % loss_)

        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        # for (b_x, b_y) in enumerate(X_train, y_train):
            # x_tmp = tf.reshape(X_train, [BATCH_SIZE, 28*28])
            # b_x, b_y = next_batch(BATCH_SIZE, x_tmp, y_train)
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
        if step % 50 == 0 or step == 599:
            print('Step:', step, '| train loss: %.4f' % loss_)
    save_path = saver.save(sess, model_path)
    print("Model saved in path: %s" % save_path)
else:
    # Restore variables from disk.
    saver.restore(sess, model_path)
    print("Model restored.")

if not NO_PLOT:
    plt.ioff()

test_output = sess.run(output, {tf_x: test_x})
pred_y = np.argmax(test_output, 1)
print(test_y.shape)
print(test_output.shape)

from sklearn import metrics
print("Classification report:\n%s\n" % (metrics.classification_report(test_y_am, pred_y)))
print("Confusion matrix:\n%s\n" % (metrics.confusion_matrix(test_y_am, pred_y)))
# print("Classification report:\n%s\n" % (metrics.classification_report(test_y, pred_y)))
# print("Confusion matrix:\n%s\n" % (metrics.confusion_matrix(test_y, pred_y)))

sess.close()