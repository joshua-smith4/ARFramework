from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import argparse
import os
from readTrafficSigns import readTrafficSigns

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--index", default=100, type=int)
parser.add_argument("--path", default=os.path.join("/home", "jsmith", "GTSRB"))
parser.add_argument("--width", type=int, default=50)
parser.add_argument("--height", type=int, default=50)

args = parser.parse_args()

x_train, y_train, x_test, y_test = readTrafficSigns(args.path,(args.width,args.height))

num_classes = len(np.unique(y_train))

x_train = x_train.astype('float32')
x_train /= 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
#y_train = y_train.astype(np.int32)

x_init = x_train[args.index:args.index+1]
y_init = y_train[args.index:args.index+1]

file_name_x = 'gtsrb_{}.pb'.format(args.index) 
file_name_y = 'gtsrb_{}_label.pb'.format(args.index) 

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

