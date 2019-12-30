import tensorflow as tf
import numpy as np

num_epochs = 2
num_class = 10
sess = tf.Session()

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


filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TextLineDataset(filenames).skip(1)
dataset = dataset.map(decode_line) # Parse the record into tensors
# print("DATASET",dataset)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
print("DATASET",dataset)
iterator = dataset.make_initializable_iterator()
features, labels = iterator.get_next()
print("ITERATOR",iterator)
print("FEATURES",features)
print("LABELS",labels)


# Initialize `iterator` with training data.
training_filenames = ["/Users/honglan/Desktop/train.csv"]
sess.run(iterator.initializer,feed_dict={filenames: training_filenames})
print("TRAIN\n",sess.run(labels))
# print(sess.run(features))

# Initialize `iterator` with validation data.
validation_filenames = ["/Users/honglan/Desktop/val.csv"]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
print("VAL\n",sess.run(labels))