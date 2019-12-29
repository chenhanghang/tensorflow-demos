import tensorflow as tf
import numpy as np

BATCH_SIZE = 4
x = np.array([[1],[2],[3],[4]])
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.repeat()

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    for _ in range(8):
        print(sess.run(el))

# [1]
# [2]
# [3]
# [4]
# [1]
# [2]
# [3]
# [4]