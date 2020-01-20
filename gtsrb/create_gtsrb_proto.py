from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--index", default=1, type=int)
args = parser.parse_args()

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
x_train /= 255

y_train = tf.keras.utils.to_categorical(y_train, 10)

x_init = x_train[args.index:args.index+1]
y_init = y_train[args.index:args.index+1]

file_name_x = 'mnist_{}.pb'.format(args.index) 
file_name_y = 'mnist_{}_label.pb'.format(args.index) 

x_proto = tf.make_tensor_proto(
        x_init, dtype=x_init.dtype, shape=x_init.shape)

y_proto = tf.make_tensor_proto(
        y_init, dtype=y_init.dtype, shape=y_init.shape)

with open(file_name_x, 'wb') as f:
    f.write(x_proto.SerializeToString())

with open(file_name_y, 'wb') as f:
    f.write(y_proto.SerializeToString())

x_proto_deserial = tf.make_tensor_proto(0)
with open(file_name_x, 'rb') as f:
    x_proto_deserial.ParseFromString(f.read())

