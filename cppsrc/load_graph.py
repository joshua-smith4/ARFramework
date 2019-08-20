import tensorflow as tf

graph = tf.Graph()
graph_def = tf.GraphDef()
with open('output_graph.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
with graph.as_default():
    tf.import_graph_def(graph_def, name='')

with tf.Session(graph=graph) as sess:
    results = sess.run(['z_out:0','oz_out:0'], feed_dict={'x_input:0': 2, 'y_input:0': 3})
print(results)
