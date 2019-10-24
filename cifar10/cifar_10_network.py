import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.keras.datasets import cifar10

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--index', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--save_train', type=int, default=1)
args = parser.parse_args()
args.save_train = bool(args.save_train)
tf.reset_default_graph()

num_classes = 10

input_layer = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_layer_x')


label_layer = tf.placeholder(tf.float32, shape=(None, 10), name='label_layer_y')

label_layer = tf.stop_gradient(label_layer, name="stopped_gradient_label")

conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv_1')

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool_1')

conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv_2')

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool_2')

conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv_3')

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name='pool_3')

flat_shape = [-1] + [np.prod(pool3.shape[1:])]
flattened = tf.reshape(pool3, flat_shape, name='flattened')

dense = tf.layers.dense(
        inputs=flattened, units=1024, activation=tf.nn.relu, name='dense_1')

logits = tf.layers.dense(inputs=dense, units=10, name='logits')
classes = tf.argmax(input=logits, axis=1, name='classes')
probabilities = tf.nn.softmax(logits, name="probabilities_out")
loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_layer, logits=logits, name='loss_func')
grad = tf.gradients(loss, input_layer)
grad_out = tf.identity(grad, name='gradient_out')

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, tf.argmax(label_layer, axis=1), 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / np.float32(255.0)
x_test = x_test / np.float32(255.0)

if args.save_train:
    x_init = x_test[args.index:args.index+1]
    y_init = y_test[args.index:args.index+1]

    file_name_x = 'cifar_{}.pb'.format(args.index) 
    file_name_y = 'cifar_{}_label.pb'.format(args.index) 

    x_proto = tf.make_tensor_proto(
            x_init, dtype=x_init.dtype, shape=x_init.shape)

    y_proto = tf.make_tensor_proto(
            y_init, dtype=y_init.dtype, shape=y_init.shape)

    with open(file_name_x, 'wb') as f:
        f.write(x_proto.SerializeToString())

    with open(file_name_y, 'wb') as f:
        f.write(y_proto.SerializeToString())

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        for i in range(x_train.shape[0] // args.batch_size):
            batch_indices = np.random.randint(
                    x_train.shape[0], size=args.batch_size)
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            sess.run(train_op, feed_dict={
                input_layer: x_batch, label_layer: y_batch})
        acc_test = accuracy.eval(feed_dict={
            input_layer: x_test, label_layer: y_test})
        #acc_train = accuracy.eval(feed_dict={
            #input_layer: x_train, label_layer: y_train})
        print("Test accuracy:", acc_test)
        #print("Train accuracy:", acc_train)

    constant_graph = graph_util.convert_variables_to_constants(
            sess, 
            sess.graph.as_graph_def(), 
            ['probabilities_out', 'gradient_out'])

    graph_io.write_graph(constant_graph, '.', 'cifar_gradient.pb', as_text=False)

