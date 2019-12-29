"""
在这里记录一下比较难理解的几个方法的用法，以便后面用到
tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, name=None)
除去name参数用以指定该操作的name，与方法有关的一共四个参数：
第一个参数sparse_indices：稀疏矩阵中那些个别元素对应的索引值。
     有三种情况：
     sparse_indices是个数，那么它只能指定一维矩阵的某一个元素
      sparse_indices是个向量，那么它可以指定一维矩阵的多个元素
     sparse_indices是个矩阵，那么它可以指定二维矩阵的多个元素
第二个参数output_shape：输出的稀疏矩阵的shape
第三个参数sparse_values：个别元素的值。
     分为两种情况：
     sparse_values是个数：所有索引指定的位置都用这个数
     sparse_values是个向量：输出矩阵的某一行向量里某一行对应的数（所以这里向量的长度应该和输出矩阵的行数对应，不然报错）
第四个参数default_value：未指定元素的默认值，一般如果是稀疏矩阵的话就是0了
"""

import tensorflow as tf
import numpy

BATCHSIZE = 6
label = tf.expand_dims(tf.constant([0, 2, 3, 6, 7, 9]), 1)
index = tf.expand_dims(tf.range(0, BATCHSIZE), 1)
# use a matrix
concated = tf.concat([index, label], 1)
onehot_labels = tf.sparse_to_dense(concated, tf.stack([BATCHSIZE, 10]), 1.0, 0.0)

# use a vector
concated2 = tf.constant([1, 3, 4])
# onehot_labels2 = tf.sparse_to_dense(concated2, tf.pack([BATCHSIZE,10]), 1.0, 0.0)#cant use ,because output_shape is not a vector
onehot_labels2 = tf.sparse_to_dense(concated2, tf.stack([10]), 1.0, 0.0)  # can use

# use a scalar
concated3 = tf.constant(5)
onehot_labels3 = tf.sparse_to_dense(concated3, tf.stack([10]), 1.0, 0.0)

with tf.Session() as sess:
    result1 = sess.run(onehot_labels)
    result2 = sess.run(onehot_labels2)
    result3 = sess.run(onehot_labels3)
    print("This is result1:")
    print(result1)
    print("This is result2:")
    print(result2)
    print("This is result3:")
    print(result3)


"""
This is result1:
[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
This is result2:
[0. 1. 0. 1. 1. 0. 0. 0. 0. 0.]
This is result3:
[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
"""