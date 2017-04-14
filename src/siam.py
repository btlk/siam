from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Flatten
import keras.optimizers as opts
from keras import backend as K
import tensorflow as tf
import cv2
import os, sys
import json


IMAGE_PIXELS = 48 * 48 * 3
IMAGE_SHAPE = [48, 48, 3]
NUM_CLASSES = 31
num_epochs = 50
batch_size = 32
num_train = 9158#22466 #9158
num_test = 7551 #2935

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
K.set_session(sess)


def read_data(images_dir, labels_file):
  images_data = []
  labels_list = [int(x.strip()) for x in open(labels_file, 'r').readlines()]

  images_list = sorted(os.listdir(images_dir))
  for im in images_list:
    with open(os.path.join(images_dir, im), 'rb') as img_stream:
      file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
      img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
      images_data.append(img_data_ndarray)

  return np.asarray(images_data), np.asarray(labels_list)


def read_data2(images_dir):
  dirs_list = os.listdir(images_dir)

  images_data = []
  labels_list = []

  for d in dirs_list:
    images_list = os.listdir(os.path.join(images_dir, d))
    for im in images_list:
      with open(os.path.join(images_dir, d, im), 'rb') as img_stream:
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        images_data.append(img_data_ndarray)
        labels_list.append(int(d))

  return np.asarray(images_data), np.asarray(labels_list)


def convert_to_npz(images, labels, output_path):
  np.savez_compressed(output_path, images = images, labels = labels)


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
  return K.mean(y_true * K.square(y_pred) 
    + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
  '''Positive and negative pair creation.
  Alternates between positive and negative pairs.
  '''
  pairs = []
  labels = []
  n = min([len(digit_indices[d]) for d in range(NUM_CLASSES)]) - 1
  for d in range(NUM_CLASSES):
    for i in range(n):
      z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
      pairs += [[x[z1], x[z2]]]
      inc = random.randrange(1, NUM_CLASSES - 1)
      dn = (d + inc) % NUM_CLASSES
      z1, z2 = digit_indices[d][i], digit_indices[dn][i]
      pairs += [[x[z1], x[z2]]]
      labels += [1, 0]
  return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
  '''Base network to be shared (eq. to feature extraction).
  '''
  with tf.device('/gpu:0'):
    seq = Sequential()
    seq.add(Conv2D(
      filters = 128, 
      kernel_size = (7, 7), 
      input_shape = input_dim, 
      activation = 'relu'))
    seq.add(MaxPooling2D())
    seq.add(Conv2D(
      filters = 128, 
      kernel_size = (4, 4), 
      activation = 'relu'))
    seq.add(MaxPooling2D())
    seq.add(Conv2D(
      filters = 256, 
      kernel_size = (3, 3), 
      activation = 'relu'))
    seq.add(Flatten())
    seq.add(Dense(
      2048, 
      activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(
      128, 
      activation='relu'))

    return seq


def dump_network(network, name = 'network'):
  # serialize model to JSON
  network_json = network.to_json()
  with open('%s.json' % name, 'w') as json_file:
      json_file.write(network_json)
  # serialize weights to HDF5
  network.save_weights('%s.h5' % name)
  print('Saved network to disk')


def load_network(name = 'network'):
  # load json and create model
  json_file = open('%s.json' % name, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights('%s.h5' % name)
  print('Loaded network from disk')

  return loaded_model


def compute_accuracy(predictions, labels):
  '''Compute classification accuracy with a fixed threshold on distances.
  '''
  return labels[predictions.ravel() < 0.5].mean()


def train_siamese(name):
  # images, labels = read_data2(r'D:\Job\misis\svn\siam\testdata\rtsd-r1\train_proc_all')
  # convert_to_npz(images, labels, r'D:\Job\misis\svn\siam\testdata\rtsd-r1\train_proc_all.npz')

  X_train, y_train = None, None
  with np.load(r'D:\Job\misis\svn\siam\testdata\rtsd-r1\train_proc_all.npz') as npf:
    X_train = npf['images'].astype(np.float32)
    y_train = npf['labels']

  X_train /= 255

  # create training+test positive and negative pairs
  digit_indices = [np.where(y_train == i)[0] for i in range(NUM_CLASSES)]
  tr_pairs, tr_y = create_pairs(X_train, digit_indices)

  # digit_indices = [np.where(y_train == i)[0] for i in range(NUM_CLASSES)]
  # te_pairs, te_y = create_pairs(X_train, digit_indices)

  # digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
  # te_pairs, te_y = create_pairs(X_test, digit_indices)

  # network definition
  with tf.device('/gpu:0'):
    base_network = create_base_network(IMAGE_SHAPE)

    input_a = Input(shape=IMAGE_SHAPE)
    input_b = Input(shape=IMAGE_SHAPE)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    # train
    opt = opts.Adam()
    model.compile(loss=contrastive_loss, optimizer=opt)
    # model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
    #           validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
    #           batch_size=batch_size,
    #           nb_epoch=num_epochs)

    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=batch_size,
              epochs=num_epochs)

    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    # pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    # te_acc = compute_accuracy(pred, te_y)

    print('* Accuracy on training set: %0.4f%%' % (100 * tr_acc))
    # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    dump_network(base_network, name)


def check_quality(name):
  X_train, y_train = None, None
  with np.load(r'D:\Job\misis\svn\siam\testdata\rtsd-r1\train_proc_all.npz') as npf:
    X_train = npf['images'].astype(np.float32)
    y_train = npf['labels']

  X_train /= 255

  digit_indices = [np.where(y_train == i)[0] for i in range(NUM_CLASSES)]
  tr_pairs, tr_y = create_pairs(X_train, digit_indices)

  with tf.device('/gpu:0'):
    base_network = load_network(name)
    input_a = Input(shape=IMAGE_SHAPE)
    input_b = Input(shape=IMAGE_SHAPE)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    opt = opts.Adam()
    model.compile(loss=contrastive_loss, optimizer=opt)

    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)

    print('* Accuracy on training set: %0.4f%%' % (100 * tr_acc))


def output_predictions(name):
  X_train, y_train = None, None
  with np.load(r'D:\Job\misis\svn\siam\testdata\rtsd-r1\train_proc_all.npz') as npf:
    X_train = npf['images'].astype(np.float32)
    y_train = npf['labels']

  X_train /= 255 # normalize

  with tf.device('/gpu:0'):
    model = load_network(name)
    model.compile(loss = 'mse', optimizer = 'adam')

    pred = model.predict(X_train)
    
    np.savetxt("predictions.csv", pred, fmt = '%.9f', delimiter = ',')
    np.savetxt("labels.csv", y_train, fmt = '%d')

if __name__ == '__main__':
  check_quality('siam_base_bak')


__all__ = ['read_data', 'read_data2', 'convert_to_npz', 
  'eucl_dist_output_shape', 'contrastive_loss', 'create_pairs', 
  'create_base_network', 'dump_network', 'load_network',
  'compute_accuracy', 'train_siamese', 'check_quality',
  'output_predictions', 'euclidean_distance']