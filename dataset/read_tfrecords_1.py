import tensorflow as tf
import numpy as np
# Validate tf.data.TFRecordDataset() using make_one_shot_iterator()
num_epochs = 2
num_class = 10
sess = tf.Session()


# Use `tf.parse_single_example()` to extract data from a `tf.Example`
# protocol buffer, and perform any additional per-record preprocessing.
def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
        "pixels": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Parse the string into an array of pixels corresponding to the image
    images = tf.decode_raw(parsed["image_raw"], tf.uint8)
    images = tf.reshape(images, [28, 28, 1])
    labels = tf.cast(parsed['label'], tf.int32)
    labels = tf.one_hot(labels, num_class)
    pixels = tf.cast(parsed['pixels'], tf.int32)
    print("IMAGES", images)
    print("LABELS", labels)

    return {"image_raw": images}, labels


filenames = ["/Users/honglan/Desktop/train_output.tfrecords"]
# replace the filenames with your own path
dataset = tf.data.TFRecordDataset(filenames)
print("DATASET", dataset)

# Use `Dataset.map()` to build a pair of a feature dictionary and a label
# tensor for each example.
dataset = dataset.map(parser)
print("DATASET_1", dataset)
dataset = dataset.shuffle(buffer_size=10000)
print("DATASET_2", dataset)
dataset = dataset.batch(32)
print("DATASET_3", dataset)
dataset = dataset.repeat(num_epochs)
print("DATASET_4", dataset)
iterator = dataset.make_one_shot_iterator()

# `features` is a dictionary in which each value is a batch of values for
# that feature; `labels` is a batch of labels.
features, labels = iterator.get_next()

print("FEATURES", features)
print("LABELS", labels)
print("SESS_RUN_LABELS \n", sess.run(labels))