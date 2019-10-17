import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.keras.datasets import cifar10

tf.reset_default_graph()

num_classes = 10
epochs = 10

input_layer = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_layer_x')
label_layer = tf.placeholder(tf.float32, shape=(None, 10), name='label_layer_y')

conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv_1')

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool_1')

conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv_2')

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool_2')

flattened = tf.reshape(pool2, [-1, 8*8*32], name='flattened')

dense = tf.layers.dense(
        inputs=flattened, units=1024, activation=tf.nn.relu, name='dense_1')

dropout = tf.nn.dropout(dense, rate=0.4, name='dropout_1')

logits = tf.layers.dense(inputs=dropout, units=10, name='logits')
classes = tf.argmax(input=logits, axis=1, name='classes')
probabilities = tf.nn.softmax(logits, name="probabilities_out")
loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_layer, logits=logits, name='loss_func')
grad = tf.gradients(loss, input_layer)
grad_out = tf.identity(grad, name='gradient_out')

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, tf.argmax(y, axis=1), 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train \= 255.0
x_test \= 255.0



