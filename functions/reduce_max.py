import tensorflow as tf
max_value1 = tf.reduce_max([[1, 3, 2],[1, 3, 4]], 1)
max_value2 = tf.reduce_max([[1, 3, 2],[1, 3, 4]], 1, True)
max_value3 = tf.reduce_max([[1, 3, 2],[1, 3, 4]], 0)
max_value4 = tf.reduce_max([[1, 3, 2],[1, 3, 4]], 0, True)
with tf.Session() as sess:
    print(sess.run(max_value1))
    print(sess.run(max_value2))
    print(sess.run(max_value3))
    print(sess.run(max_value4))