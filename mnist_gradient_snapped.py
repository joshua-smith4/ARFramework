import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from snapGrid import snapGrid

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(None, 28, 28), name='x_input')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y_label')
y = tf.stop_gradient(y, name="stop_gradient_y")

input_layer = tf.reshape(x, [-1, 28, 28, 1], name='x_reshaped')
conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv_1')

pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=2, name='pool_1')

conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv_2')

pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=2, name='pool_2')

pool2_flat = tf.reshape(
        pool2, [-1, 7 * 7 * 64], name="pool_2_flat")

dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu, name='dense_1')

logits = tf.layers.dense(inputs=dense, units=10, name='logits')
classes = tf.argmax(input=logits, axis=1, name='classes')
probabilities = tf.nn.softmax(logits, name="probabilities_out")
loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=logits, name='loss_func')
grad = tf.gradients(loss, x)
grad_out = tf.identity(grad, name='gradient_out')

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, tf.argmax(y, axis=1), 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/np.float32(255)
y_train = y_train.astype(np.int32)
x_test = x_test/np.float32(255)
y_test = y_test.astype(np.int32)

num_epsilons = 3
x_test_snapped = np.empty_like(x_test)
ref_point = np.zeros(x_test[0].shape)
granularity = np.full(x_test[0].shape, 1/256*num_epsilons)
print('Snapping X test')
for i in range(len(x_test)):
    x_test_snapped[i] = snapGrid(x_test[i], ref_point, granularity)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

num_epochs = 10
batch_size = 100

init = tf.global_variables_initializer()

print('Training')
with tf.Session() as sess:
    init.run()
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        for i in range(x_train.shape[0] // batch_size):
            batch_indices = np.random.randint(
                    x_train.shape[0], size=batch_size)
            x_batch = x_train[batch_indices]
            x_snapped = np.empty_like(x_batch)
            for j in range(batch_size):
                x_snapped[j] = snapGrid(
                        x_batch[j], 
                        ref_point,
                        granularity)
            y_batch = y_train[batch_indices]
            sess.run(train_op, feed_dict={x: x_snapped, y: y_batch})
        acc_test = accuracy.eval(feed_dict={x: x_test_snapped, y: y_test})
        print(epoch, "Test accuracy:", acc_test)

    constant_graph = graph_util.convert_variables_to_constants(
            sess, 
            sess.graph.as_graph_def(), 
            ['probabilities_out', 'gradient_out'])

    graph_io.write_graph(constant_graph, '.', 'mnist_gradient.pb', as_text=False)


