from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import argparse
import os
from readTrafficSigns import readTrafficSigns

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=os.path.join("/home", "jsmith", "GTSRB"))
args = parser.parse_args()

x_train, y_train, x_test, y_test = readTrafficSigns(args.path)
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]

input_shape = (img_rows, img_cols, 3)

num_classes = len(np.unique(y_train))

x_train = x_train.astype('float32')
x_train /= 255.0
counts = np.zeros(num_classes)
avg = np.zeros((num_classes,)+x_train[0].shape, dtype=np.float32)
for i in range(len(y_train)):
    counts[y_train[i]] += 1
    avg[y_train[i]] += x_train[i]

for i in range(len(counts)):
    avg[i] /= counts[i]

file_name = 'gtsrb_averages.pb'
avg_proto = tf.make_tensor_proto(avg, dtype=avg.dtype, shape=avg.shape)

with open(file_name, 'wb') as f:
    f.write(avg_proto.SerializeToString())

