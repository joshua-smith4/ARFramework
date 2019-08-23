from keras.datasets import mnist
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows = 28
img_cols = 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

index = 1000
x_init = x_train[index:index+1]
print(y_train[index])

import tensorflow as tf

file_name = 'mnist_{}.pb'.format(index) 
x_proto = tf.make_tensor_proto(x_init, dtype=x_init.dtype, shape=x_init.shape)
with open(file_name, 'wb') as f:
    f.write(x_proto.SerializeToString())

x_proto_deserial = tf.make_tensor_proto(0)
with open(file_name, 'rb') as f:
    x_proto_deserial.ParseFromString(f.read())

print(x_proto_deserial)

x = tf.convert_to_tensor(x_proto_deserial)

with tf.Session() as sess:
    print(x.eval(session=sess))
