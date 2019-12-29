import tensorflow as tf
import numpy as np

BATCH_SIZE = 4
x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))

# [[0.70861276 0.91522017]
#  [0.993154   0.74425373]
#  [0.42730845 0.03037355]
#  [0.54031161 0.57429001]]