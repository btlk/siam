import numpy as np
import random
from argparse import ArgumentParser
from keras.models import Model
from keras.layers import Input, Lambda
from keras.utils import plot_model
from keras import optimizers as opts
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import os, shutil

from data import load_npz
from net import create_base_network


batch_size = 128


config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)


def euclidean_distance(vects):
  x, y = vects
  return K.sqrt(K.sum(K.square(x - y), axis = 1, keepdims = True))


def eucl_dist_output_shape(shapes):
  shape1, shape2 = shapes
  return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
  '''Contrastive loss from Hadsell-et-al.'06
  http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
  '''
  margin = 1
  return K.mean(y_true * K.square(y_pred) 
    + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices, num_classes):
  '''Positive and negative pair creation.
  Alternates between positive and negative pairs.
  '''
  pairs = []
  labels = []
  n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
  for d in range(num_classes):
    for i in range(n):
      inds = np.random.randint(0, len(digit_indices[d]), 2)
      z1, z2 = digit_indices[d][inds[0]], digit_indices[d][inds[1]]
      pairs += [[x[z1], x[z2]]]
      inc = random.randrange(1, num_classes - 1)
      dn = (d + inc) % num_classes
      indn = np.random.randint(0, len(digit_indices[dn]))
      if np.random.rand() < 0.5:
        indi = inds[0]
      else:
        indi = inds[1]
      z1, z2 = digit_indices[d][indi], digit_indices[dn][indn]
      pairs += [[x[z1], x[z2]]]
      labels += [1, 0]
  return np.array(pairs), np.array(labels)


def compute_accuracy(predictions, labels):
  return labels[predictions.ravel() < 0.5].mean()


def train_siamese(X_train, y_train, X_test, y_test, output_path, epochs):
  image_shape = X_train[0].shape
  num_classes = y_train.max() + 1
  digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
  tr_pairs, tr_y = create_pairs(X_train, digit_indices, num_classes)

  digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
  te_pairs, te_y = create_pairs(X_test, digit_indices, num_classes)

  # network definition
  with tf.device('/gpu:0'):
    base_network = create_base_network(image_shape)
    input_a = Input(shape = image_shape)
    input_b = Input(shape = image_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(
      euclidean_distance, 
      output_shape = eucl_dist_output_shape
    )([processed_a, processed_b])

    model = Model(
      inputs = [input_a, input_b], 
      outputs = distance
    )

    opt = opts.Adam()
    model.compile(
      loss = contrastive_loss, 
      optimizer = opt
    )

    callbacks_list = []
    if output_path is not None:
      if os.path.exists(output_path):
        shutil.rmtree(output_path)
      os.makedirs(output_path)
      filepath = os.path.join(output_path, '{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
      checkpoint = ModelCheckpoint(
        filepath, 
        monitor = 'val_loss', 
        verbose = 0, 
        save_best_only = True, 
        mode = 'min'
      )
      callbacks_list = [checkpoint]

    history = model.fit(
      x = [tr_pairs[:, 0], tr_pairs[:, 1]], 
      y = tr_y,
      callbacks = callbacks_list,
      batch_size = batch_size,
      validation_data = ([te_pairs[:, 0], te_pairs[:, 1]], te_y),
      epochs = epochs
    )

    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)

    print('* Accuracy on training set: %0.4f%%' % (100 * tr_acc))


__all__ = ['euclidean_distance', 'eucl_dist_output_shape', 'contrastive_loss', 
  'create_pairs', 'compute_accuracy', 'train_siamese']


if __name__ == '__main__':
  parser = ArgumentParser(description = 'Data runner')
  parser.add_argument('--train-data', '-t',
                      type = str,
                      help = 'Path to train data',
                      required = True)
  parser.add_argument('--validation-data', '-v',
                      type = str,
                      help = 'Path to validation data',
                      required = True)
  parser.add_argument('--output-path', '-o',
                      type = str,
                      help = 'Path to output dir',
                      default = None)
  parser.add_argument('--epochs', '-e',
                      type = int,
                      help = 'Epochs count',
                      default = 50)
  args = parser.parse_args()
  train_path = args.train_data
  val_path = args.validation_data
  output_path = args.output_path
  epochs = args.epochs
  X_train, y_train = load_npz(train_path)
  X_val, y_val = load_npz(val_path)
  X_train = X_train.astype(np.float32) / 255.0
  X_val = X_val.astype(np.float32) / 255.0
  train_siamese(X_train, y_train, X_val, y_val, output_path, epochs)