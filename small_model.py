import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

tf.reset_default_graph()

x = tf.placeholder(tf.float32, name='x_input')
y = tf.placeholder(tf.float32, name='y_input')

z = tf.add(tf.multiply(y,y), x, name='z_out')
oz = tf.add(y,x, name='oz_out')

with tf.Session() as sess:
    const_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['z_out','oz_out'])
    graph_io.write_graph(const_graph, '.', 'output_graph.pb', as_text=False)
