import tensorflow as tf
import numpy as np

num_epochs = 2
num_class = 10
sess = tf.Session()
# validate tf.data.TextLineDataset() using Reinitializable iterator
# In order to switch between train and validation data

def decode_line(line):
    # Decode the csv_line to tensor.
    record_defaults = [[1.0] for col in range(785)]
    items = tf.decode_csv(line, record_defaults)
    features = items[1:785]
    label = items[0]

    features = tf.cast(features, tf.float32)
    features = tf.reshape(features,[28,28,1])
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

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
features, labels = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Using reinitializable iterator to alternate between training and validation.
sess.run(training_init_op)
print("TRAIN\n",sess.run(labels))
# print(sess.run(features))

# Reinitialize `iterator` with validation data.
sess.run(validation_init_op)
print("VAL\n",sess.run(labels))