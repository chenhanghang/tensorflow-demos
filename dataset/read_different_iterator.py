import tensorflow as tf
import numpy as np

num_epochs = 2
num_class = 10
sess = tf.Session()

# validate tf.data.TextLineDataset() using two different iterator
# In order to switch between train and validation data

def decode_line(line):
    # Decode the csv_line to tensor.
    record_defaults = [[1.0] for col in range(785)]
    items = tf.decode_csv(line, record_defaults)
    features = items[1:785]
    label = items[0]

    features = tf.cast(features, tf.float32)
    features = tf.reshape(features,[28,28])
    label = tf.cast(label, tf.int64)
    label = tf.one_hot(label,num_class)
    return features,label


def create_dataset(filename, batch_size=32, is_shuffle=False, n_repeats=0):
    """create dataset for train and validation dataset"""
    dataset = tf.data.TextLineDataset(filename).skip(1)
    if n_repeats > 0:
        dataset = dataset.repeat(n_repeats)         # for train
    # dataset = dataset.map(decode_line).map(normalize)
    dataset = dataset.map(decode_line)
    # decode and normalize
    if is_shuffle:
        dataset = dataset.shuffle(10000)            # shuffle
    dataset = dataset.batch(batch_size)
    return dataset


training_filenames = ["/Users/honglan/Desktop/train.csv"]
# replace the filenames with your own path
validation_filenames = ["/Users/honglan/Desktop/val.csv"]
# replace the filenames with your own path

# Create different datasets
training_dataset = create_dataset(training_filenames, batch_size=32, \
                                  is_shuffle=True, n_repeats=num_epochs) # train_filename
validation_dataset = create_dataset(validation_filenames, batch_size=32, \
                                  is_shuffle=True, n_repeats=num_epochs) # val_filename

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
features, labels = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Using different handle to alternate between training and validation.
print("TRAIN\n",sess.run(labels, feed_dict={handle: training_handle}))
# print(sess.run(features))

# Initialize `iterator` with validation data.
sess.run(validation_iterator.initializer)
print("VAL\n",sess.run(labels, feed_dict={handle: validation_handle}))