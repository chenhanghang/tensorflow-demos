"""
Concatenates tensors along one dimension.
"""
import tensorflow as tf
import numpy

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
s1=tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
s2=tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
s3=tf.concat([t1, t2], -1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

with tf.Session() as sess:
  print(sess.run(s1))
  print(sess.run(s2))
  print(sess.run(s3))