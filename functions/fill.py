"""
  # Output tensor has shape [2, 3].
  fill([2, 3], 9) ==> [[9, 9, 9]
                       [9, 9, 9]]
"""
import tensorflow as tf
t1 = tf.fill([2,3,4], 3)
sess = tf.Session()

print(sess.run(t1))