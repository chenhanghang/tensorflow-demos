import tensorflow as tf
import numpy as np

tf.reset_default_graph()

x = [[1, 0, 0],
     [2, 3, 0],
     [4, 5, 6],
     [7, 8, 0],
     [9, 0, 0],
     [0, 1, 0]]

# tf.TensorShape([])     表示长度为单个数字
# tf.TensorShape([None]) 表示长度未知的向量
padded_shapes = (
    tf.TensorShape([]),
    tf.TensorShape([]),
    tf.TensorShape([])
)

dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.map(lambda x: [x[0], x[1], x[2]])
dataset = dataset.padded_batch(2, padded_shapes=padded_shapes)
iterator = dataset.make_one_shot_iterator()
sess = tf.Session()
try:
    while True:
        elem1, elem2, elem3 = iterator.get_next()
        print("elem:", sess.run(elem1))
except tf.errors.OutOfRangeError:
    print("end")
"""
elem: [1 2]
elem: [4 7]
elem: [9 0]
"""