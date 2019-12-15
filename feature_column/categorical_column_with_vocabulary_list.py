import tensorflow as tf

pets = {'pets': [['rabbit', "pig"],['pig', "pig"],['dog', "pig"],['mouse', "pig"],['cat', "pig"]]}

#猫狗兔子猪4个+num_oov_buckets =7维度
column = tf.feature_column.categorical_column_with_vocabulary_list(
    key='pets',
    vocabulary_list=['cat','dog','rabbit','pig'],
    dtype=tf.string,
    default_value=-1,
    num_oov_buckets=3)

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(pets, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))

    # [array([[0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 1., 0., 0., 0.],
    #         [0., 1., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 1., 0.],
    #         [1., 0., 0., 0., 0., 0., 0.]], dtype=float32)]