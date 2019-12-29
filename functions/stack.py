"""
tf.stack(values, axis=0, name=”pack”)
Packs a list of rank-R tensors into one rank-(R+1) tensor
将一个R维张量列表沿着axis轴组合成一个R+1维的张量
"""
import tensorflow as tf
import numpy
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])

pack1 = tf.stack([x, y, z])  #[[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
pack2 = tf.stack([x, y, z], axis=1) #[[1, 2, 3], [4, 5, 6]]

with tf.Session() as sess:
  print(sess.run(pack1))
  print(sess.run(pack2))