import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.keras.datasets import cifar10

tf.reset_default_graph()

num_classes = 10

input_layer = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_layer_x')

input_layer = tf.stop_gradient(input_layer, name="stopped_gradient_input")

label_layer = tf.placeholder(tf.float32, shape=(None, 10), name='label_layer_y')

label_layer = tf.stop_gradient(label_layer, name="stopped_gradient_label")

conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv_1')

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool_1')

conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv_2')

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool_2')

print(pool2.shape)
flattened = tf.reshape(pool2, [-1, 8*8*64], name='flattened')

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
correct = tf.nn.in_top_k(logits, tf.argmax(label_layer, axis=1), 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print(y_train.shape, y_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255.0
x_test /= 255.0

index = 100

x_init = x_test[index:index+1]
y_init = y_test[index:index+1]

file_name_x = 'cifar_{}.pb'.format(index) 
file_name_y = 'cifar_{}_label.pb'.format(index) 

x_proto = tf.make_tensor_proto(
        x_init, dtype=x_init.dtype, shape=x_init.shape)

y_proto = tf.make_tensor_proto(
        y_init, dtype=y_init.dtype, shape=y_init.shape)

with open(file_name_x, 'wb') as f:
    f.write(x_proto.SerializeToString())

with open(file_name_y, 'wb') as f:
    f.write(y_proto.SerializeToString())

epochs = 10
batch_size = 100

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        for i in range(x_train.shape[0] // batch_size):
            batch_indices = np.random.randint(
                    x_train.shape[0], size=batch_size)
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            sess.run(train_op, feed_dict={
                input_layer: x_batch, label_layer: y_batch})
        acc_test = accuracy.eval(feed_dict={
            input_layer: x_test, label_layer: y_test})
        print(epoch, "Test accuracy:", acc_test)

    constant_graph = graph_util.convert_variables_to_constants(
            sess, 
            sess.graph.as_graph_def(), 
            ['probabilities_out', 'gradient_out'])

    graph_io.write_graph(constant_graph, '.', 'cifar_gradient.pb', as_text=False)

