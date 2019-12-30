"""
tf.nn.embedding_lookup_sparse(
params,
sp_ids,
sp_weights,
partition_strategy='mod',
name=None,
combiner=None,
max_norm=None
)

params embedding使用的lookup table.
sp_ids 查找lookup table的SparseTensor.
combiner 通过什么运算把一行的数据结合起来mean, sum等.
"""

import numpy as np
import tensorflow as tf

### embedding matrix
example = np.arange(24).reshape(6, 4).astype(np.float32)
embedding = tf.Variable(example)

"""
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.],
       [12., 13., 14., 15.],
       [16., 17., 18., 19.],
       [20., 21., 22., 23.]], dtype=float32)
"""

### embedding lookup SparseTensor
idx = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 1], [1, 2], [2, 0]],
                      values=[0, 1, 2, 3, 0], dense_shape=[3, 3])

"""
array([[0, 1, None],
       [None, 2, 3],
       [0, None, None]]) # 为了与0元素相区别，没有填充的部分写成了None
"""

embed = tf.nn.embedding_lookup_sparse(embedding, idx, None, combiner='sum')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(embed))
