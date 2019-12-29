import tensorflow as tf
import numpy as np

x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))

# tuple 处理
features, labels = (np.random.sample((100,2)), np.random.sample((100,1)))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))

# use tensor
dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run(el))


# using a placeholder
x = tf.placeholder(tf.float32, shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices(x)

data = np.random.sample((100,2))

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer, feed_dict={ x: data })
    print(sess.run(el))

# from generator
sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])

def generator():
    for el in sequence:
        yield el

dataset = tf.data.Dataset().batch(1).from_generator(generator,
                                           output_types= tf.int64,
                                           output_shapes=(tf.TensorShape([None, 1])))

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))

# feedable iterator to switch between iterators
EPOCHS = 10
# making fake data using numpy
train_data = (np.random.sample((100, 2)), np.random.sample((100, 1)))
test_data = (np.random.sample((10, 2)), np.random.sample((10, 1)))
# create placeholder
x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])
# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
test_dataset = tf.data.Dataset.from_tensor_slices((x, y))
# create the iterators from the dataset
train_iterator = train_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()
# same as in the doc https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
handle = tf.placeholder(tf.string, shape=[])
iter = tf.data.Iterator.from_string_handle(
    handle, train_dataset.output_types, train_dataset.output_shapes)
next_elements = iter.get_next()

with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    # initialise iterators. In our case we could have used the 'one-shot' iterator instead,
    # and directly feed the data insted the Dataset.from_tensor_slices function, but this
    # approach is more general
    sess.run(train_iterator.initializer, feed_dict={x: train_data[0], y: train_data[1]})
    sess.run(test_iterator.initializer, feed_dict={x: test_data[0], y: test_data[1]})

    for _ in range(EPOCHS):
        x, y = sess.run(next_elements, feed_dict={handle: train_handle})
        print(x, y)

    x, y = sess.run(next_elements, feed_dict={handle: test_handle})
    print(x, y)
# [0.8552025  0.13344285][0.24534453]
# [0.23880187 0.2294315][0.77315474]
# [0.763904 0.439595][0.42727667]
# [0.6563372 0.1366187][0.02278621]
# [0.71135175 0.394754][0.8552778]
# [0.7329701  0.42924434][0.43608633]
# [0.8240853 0.7750715][0.5140434]
# [0.65556693 0.67978406][0.8228361]
# [0.02365288 0.18461536][0.85140544]
# [0.48037764 0.7320316][0.773141]
# [0.6671238 0.8491173][0.45188755]