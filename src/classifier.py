import numpy as np
import random
from argparse import ArgumentParser
from keras.models import Model
from keras.layers import Input, Lambda
from keras.utils import plot_model, to_categorical
from keras import optimizers as opts
from keras import backend as K
import tensorflow as tf
import os, shutil

from data import load_npz
from net import (attach_classifier, create_base_network, 
  SeparateSaveCallback)
from siam import contrastive_loss

# workaround
from keras import losses
losses.contrastive_loss = contrastive_loss
# ---------- #


batch_size = 128

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)


def acc(y_true, y_pred):
  correct = tf.equal(
    tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))


def train_classifier(X_train, y_train, 
  X_test, y_test, model_path, output_path, epochs):
  image_shape = X_train[0].shape
  model = create_base_network(X_train.shape[1:])
  model.load_weights(model_path)
  model = attach_classifier(model, 2)

  opt = opts.RMSprop(epsilon = 1e-4, decay = 1e-6)
  model.compile(
    loss = 'categorical_crossentropy', 
    metrics = [acc],
    optimizer = opt
  )

  callbacks_list = []
  if output_path is not None:
    if os.path.exists(output_path):
      shutil.rmtree(output_path)
    os.makedirs(output_path)
    file_fmt = '{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
    checkpoint = SeparateSaveCallback(
      output_path, file_fmt, siamese = False)
    callbacks_list = [checkpoint]

  y_train = to_categorical(y_train, 2)
  if X_test is not None and y_test is not None:
    y_test = to_categorical(y_test, 2)
    history = model.fit(
      x = X_train, 
      y = y_train,
      callbacks = callbacks_list,
      batch_size = batch_size,
      validation_data = (X_test, y_test),
      epochs = epochs
    )

  else:
    history = model.fit(
      x = X_train, 
      y = y_train,
      callbacks = callbacks_list,
      batch_size = batch_size,
      validation_split = 0.5,
      epochs = epochs
    )

  return history

__all__ = ['train_classifier']


if __name__ == '__main__':
  parser = ArgumentParser(description = 'Data runner')
  parser.add_argument('--train-data', '-t',
                      type = str,
                      help = 'Path to train data',
                      required = True)
  parser.add_argument('--validation-data', '-v',
                      type = str,
                      help = 'Path to validation data')
  parser.add_argument('--model', '-m',
                      type = str,
                      help = 'Path to trained siamese model',
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
  model_path = args.model
  X_train, y_train = load_npz(train_path)

  if val_path and os.path.exists(val_path):
    X_val, y_val = load_npz(val_path)
    X_val = X_val.astype(np.float32) / 255.0
  else:
    X_val, y_val = None, None

  X_train = X_train.astype(np.float32) / 255.0
  history = train_classifier(X_train, y_train, 
    X_val, y_val, model_path, output_path, epochs)

  with open('history_cl_loss.txt', 'w') as hstr:
    hstr.write(str(history.history['loss']))

  with open('history_cl_valloss.txt', 'w') as hstr:
    hstr.write(str(history.history['val_loss']))

  with open('history_cl_acc.txt', 'w') as hstr:
    hstr.write(str(history.history['acc']))

  with open('history_cl_valacc.txt', 'w') as hstr:
    hstr.write(str(history.history['val_acc']))