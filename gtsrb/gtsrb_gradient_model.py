import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from readTrafficSigns import readTrafficSigns
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--path", default=os.path.join("/home", "jsmith", "GTSRB"))
args = parser.parse_args()

tf.reset_default_graph()

x_train, y_train, x_test, y_test = readTrafficSigns(args.path)

num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

input_shape = (None,) + x_train.shape[1:]
label_shape = (None, num_classes)

x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1], x_train.shape[2], x_train.shape[3]), name='input_layer_x')
y = tf.placeholder(tf.float32, shape=label_shape, name='label_layer_y')
y = tf.stop_gradient(y, name="stop_gradient_y")

conv1 = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv_1')

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool_1')

conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv_2')

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool_2')

conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv_3')

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name='pool_3')

print(pool3.shape)

flat_shape = [-1, np.prod(pool3.shape[1:])]
pool3_flat = tf.reshape(pool3, flat_shape, name="pool_3_flat")

dense1 = tf.layers.dense(
        inputs=pool3_flat, units=1024, activation=tf.nn.relu, name='dense_1')
dense2 = tf.layers.dense(
        inputs=dense1, units=512, activation=tf.nn.relu, name='dense_2')

logits = tf.layers.dense(inputs=dense2, units=num_classes, name='logits')
classes = tf.argmax(input=logits, axis=1, name='classes')
probabilities = tf.nn.softmax(logits, name="probabilities_out")
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='loss_func')
grad = tf.gradients(loss, x)
grad_out = tf.identity(grad, name='gradient_out')

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, tf.argmax(y, axis=1), 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

x_train = x_train/np.float32(255)
y_train = y_train.astype(np.int32)
x_test = x_test/np.float32(255)
y_test = y_test.astype(np.int32)

batch_size = 100

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        for i in range(x_train.shape[0] // batch_size):
            batch_indices = np.random.randint(x_train.shape[0], size=batch_size)
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            sess.run(train_op, feed_dict={x: x_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={x: x_test, y: y_test})
        print(epoch, "Test accuracy:", acc_test)

    constant_graph = graph_util.convert_variables_to_constants(
            sess, 
            sess.graph.as_graph_def(), 
            ['probabilities_out', 'gradient_out'])

    graph_io.write_graph(constant_graph, '.', 'gtsrb_gradient.pb', as_text=False)

