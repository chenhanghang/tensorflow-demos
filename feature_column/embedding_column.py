import tensorflow as tf

features = {'pets': ['dog','cat','rabbit','pig','mouse']}

pets_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    'pets',
    ['cat','dog','rabbit','pig'],
    dtype=tf.string,
    default_value=-1)

column = tf.feature_column.embedding_column(pets_f_c, 3)
indicator_column = tf.feature_column.indicator_column(pets_f_c)

indicator_tensor = tf.feature_column.input_layer(features, [indicator_column])
tensor = tf.feature_column.input_layer(features, [column])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    print(session.run([tensor]))
    print(session.run([indicator_tensor]))

    # [array([[0.6827723, 0.80465585, 0.38107032],
    #         [0.5135401, 0.02038863, 0.8029042],
    #         [0.10791357, -0.10505158, 0.14126506],
    #         [-0.03034976, -0.36720538, -0.14336108],
    #         [0., 0., 0.]], dtype=float32)]

    # [array([[0., 1., 0., 0.],
    #         [1., 0., 0., 0.],
    #         [0., 0., 1., 0.],
    #         [0., 0., 0., 1.],
    #         [0., 0., 0., 0.]], dtype=float32)]
