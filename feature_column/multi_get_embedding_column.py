import tensorflow as tf

features = {'pets': ['dog','cat','rabbit','pig','mouse'],
            'price': [2, 3, 0, 1, 1]}

pets_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    'pets',
    ['cat', 'dog', 'rabbit', 'pig'],
    dtype=tf.string,
    default_value=-1)

p_c =  tf.feature_column.categorical_column_with_identity(
    key='price',
    num_buckets=4)

pet_column = tf.feature_column.embedding_column(pets_f_c, 3)
price_column = tf.feature_column.embedding_column(p_c, 3)

tensor = tf.feature_column.input_layer(features, [price_column])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    print(session.run([tensor]))

