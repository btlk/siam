'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
import cv2
import os, sys


TRAIN_DIR = 'D:\Job\misis\svn\siam\testdata\rtsd-r1'
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'test.tfrecords'
IMAGE_PIXELS = 48 * 48 * 3


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, output_path):
  """Converts a dataset to tfrecords."""
  num_examples = len(images)

  output_dir = os.path.split(output_path)[0]
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if len(labels) != num_examples:
    raise ValueError('Images count %d does not match labels count %d.' %
                     (num_examples, len(labels)))
  rows = images[0].shape[0]
  cols = images[0].shape[1]
  depth = images[0].shape[2]

  filename = output_path
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(labels[index]),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def read_data(images_dir, labels_file):
  images_list = os.listdir(images_dir)
  labels_list = [int(x.strip()) for x in open(labels_file, 'r').readlines()] 

  assert len(images_list) == len(labels_list)
  images_data = []

  for im in images_list:
    with open(os.path.join(images_dir, im), 'rb') as img_stream:
      file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
      img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

      images_data.append(img_data_ndarray)

  return images_data, labels_list


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([IMAGE_PIXELS])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(TRAIN_DIR,
                          TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

  return images, sparse_labels


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(67)]) - 1
    for d in range(67):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 67)
            dn = (d + inc) % 67
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    with tf.device('/gpu:0'):
      seq = Sequential()
      seq.add(Conv2D(filters = 32, kernel_size = (7, 7), input_shape=(input_dim,), activation = 'relu'))
      seq.add(MaxPooling2D())
      seq.add(Conv2D(filters = 64, kernel_size = (6, 6), input_shape=(input_dim,), activation = 'relu'))
      seq.add(MaxPooling2D())
      seq.add(Dense(128, activation='relu'))
      # seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
      # seq.add(Dropout(0.1))
      # seq.add(Dense(128, activation='relu'))
      # seq.add(Dropout(0.1))
      # seq.add(Dense(128, activation='relu'))

    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


if __name__ == '__main__':
  # images, labels = read_data(r'D:\Job\misis\svn\siam\testdata\rtsd-r1\test_proc', 
  #   r'D:\Job\misis\svn\siam\testdata\rtsd-r1\gt_test.csv')
  # convert_to(images, labels, r'D:\Job\misis\svn\siam\testdata\rtsd-r1\test.tfrecords')
  # the data, shuffled and split between train and test sets
  num_epochs = 2
  input_dim = IMAGE_PIXELS
  batch_size = 32
  num_classes = 67
  (X_train, y_train) = inputs(train = True, batch_size = batch_size, num_epochs = num_epochs)
  (X_test, y_test) = inputs(train = False, batch_size = batch_size, num_epochs = num_epochs)

  # create training+test positive and negative pairs
  digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
  tr_pairs, tr_y = create_pairs(X_train, digit_indices)

  digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
  te_pairs, te_y = create_pairs(X_test, digit_indices)

  # network definition
  base_network = create_base_network(input_dim)

  input_a = Input(shape=(input_dim,))
  input_b = Input(shape=(input_dim,))

  # because we re-use the same instance `base_network`,
  # the weights of the network
  # will be shared across the two branches
  processed_a = base_network(input_a)
  processed_b = base_network(input_b)

  distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

  model = Model(input=[input_a, input_b], output=distance)

  # train
  rms = RMSprop()
  model.compile(loss=contrastive_loss, optimizer=rms)
  model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
            validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
            batch_size=batch_size,
            nb_epoch=num_epochs)

  # compute final accuracy on training and test sets
  pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
  tr_acc = compute_accuracy(pred, tr_y)
  pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
  te_acc = compute_accuracy(pred, te_y)

  print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
  print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
