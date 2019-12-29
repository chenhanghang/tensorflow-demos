import tensorflow as tf
import numpy as np

# MAP
x = np.array([[1],[2],[3],[4]])
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.map(lambda x: x*2)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
#     this will run forever
        for _ in range(len(x)):
            print(sess.run(el))