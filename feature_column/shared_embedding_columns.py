import tensorflow as tf
from tensorflow import feature_column

with tf.Session() as sess:
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed[0]))
    print(color_column_embed[0].name)
    color_dense_tensor1 = feature_column.input_layer(color_data, color_column_embed)
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_embed[0]])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(color_dense_tensor))
    print(sess.run(color_dense_tensor1))

# [[-0.25656056  0.12558194 -0.8840832  -0.5131211   0.25116387 -1.7681664 ]
#  [ 0.07877526 -0.4242465   0.68331635  0.15755053 -0.848493    1.3666327 ]
#  [ 0.          0.          0.          0.4511305  -0.15397087 -0.02873593]
#  [ 0.4511305  -0.15397087 -0.02873593  0.902261   -0.30794173 -0.05747186]]