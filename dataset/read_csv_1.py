import tensorflow as tf
import numpy as np
# validate tf.data.TextLineDataset() using make_one_shot_iterator()
num_epochs = 2
num_class = 10
sess = tf.Session()
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


filenames = ["/Users/honglan/Desktop/train.csv"]
# replace the filenames with your own path
dataset = tf.data.TextLineDataset(filenames).skip(1)
print("DATASET",dataset)

# Use `Dataset.map()` to build a pair of a feature dictionary and a label
# tensor for each example.
dataset = dataset.map(decode_line)
print("DATASET_1",dataset)
dataset = dataset.shuffle(buffer_size=10000)
print("DATASET_2",dataset)
dataset = dataset.batch(32)
print("DATASET_3",dataset)
dataset = dataset.repeat(num_epochs)
print("DATASET_4",dataset)
iterator = dataset.make_one_shot_iterator()

# `features` is a dictionary in which each value is a batch of values for
# that feature; `labels` is a batch of labels.
features, labels = iterator.get_next()

print("FEATURES",features)
print("LABELS",labels)
print("SESS_RUN_LABELS\n",sess.run(labels))