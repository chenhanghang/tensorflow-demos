import tensorflow as tf
#构建序列长度的mask标志
# lengths：整数张量，其所有值小于等于maxlen。
# maxlen：标量整数张量，返回张量的最后维度的大小；默认值是lengths中的最大值。
# dtype：结果张量的输出类型。
# name：操作的名字。
#返回一个表示每个单元的前N个位置的mask张量
# 由此产生的张量mask有dtype类型和形状[d_1, d_2, ..., d_n, maxlen]，并且：
#
# mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
mask = tf.sequence_mask([1, 3, 2], 5)



with tf.Session() as sess:
    mask = sess.run(mask)
    print(mask)

# [[ True False False False False]
#  [ True  True  True False False]
#  [ True  True False False False]]