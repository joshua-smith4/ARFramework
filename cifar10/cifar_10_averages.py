from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_rows = 32
img_cols = 32
num_classes = 10

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)


x_train = x_train.astype('float32')
x_train /= 255.0
counts = np.zeros(num_classes)
avg = np.zeros((num_classes,)+x_train[0].shape, dtype=np.float32)
for i in range(len(y_train)):
    counts[y_train[i]] += 1
    avg[y_train[i]] += x_train[i]

for i in range(len(counts)):
    avg[i] /= counts[i]

file_name = 'cifar_10_averages.pb'
avg_proto = tf.make_tensor_proto(avg, dtype=avg.dtype, shape=avg.shape)

with open(file_name, 'wb') as f:
    f.write(avg_proto.SerializeToString())

