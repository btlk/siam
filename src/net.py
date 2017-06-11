import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import (Dense, Dropout, Input, 
  Conv2D, MaxPooling2D, Flatten, GaussianNoise)
from keras.callbacks import Callback
import tensorflow as tf
import os, shutil
import json


def create_base_network(input_dim):
  seq = Sequential()
  seq.add(GaussianNoise(
    stddev = 0.05,
    input_shape = input_dim))
  seq.add(Conv2D(
    filters = 64, 
    kernel_size = (3, 3), 
    activation = 'relu'))
  seq.add(Conv2D(
    filters = 32, 
    kernel_size = (3, 3), 
    activation = 'relu'))
  seq.add(MaxPooling2D())
  seq.add(Conv2D(
    filters = 32, 
    kernel_size = (3, 3), 
    activation = 'relu'))
  seq.add(MaxPooling2D())
  seq.add(Conv2D(
    filters = 64, 
    kernel_size = (3, 3), 
    activation = 'relu'))
  seq.add(Conv2D(
    filters = 8, 
    kernel_size = (1, 1), 
    activation = 'relu'))
  seq.add(Flatten())
  seq.add(Dense(
    256, 
    activation='relu'))
  seq.add(Dropout(0.5))
  seq.add(Dense(
    32, 
    activation='relu'))

  return seq


def attach_classifier(model, output_pins):
  for l in model.layers:
    l.trainable = False
  model.add(Dense(
    output_pins,
    activation='softmax',
    name = 'dense_clas_1'
  ))
  return model


class SeparateSaveCallback(Callback):
  def __init__(self, save_dir, file_fmt, siamese = True):
    super(SeparateSaveCallback, self).__init__()
    self.save_dir = save_dir
    self.file_fmt = file_fmt
    self.siamese = siamese

  def on_train_begin(self, logs = None):
    json_model = None
    if self.siamese:
      json_model = self.model.layers[2].to_json()
    else:
      json_model = self.model.to_json()
    with open(os.path.join(self.save_dir, 
      'model.json'), 'w') as json_file:
      json_file.write(json_model)

  def on_train_end(self, logs = None):
    pass

  def on_epoch_begin(self, epoch, logs = None):
    pass

  def on_epoch_end(self, epoch, logs = None):
    logs = logs or {}
    if self.siamese:
      self.model.layers[2].save_weights(
        os.path.join(self.save_dir, 
          self.file_fmt.format(
            epoch = epoch, **logs)),
        overwrite = True)
    else:
      self.model.save_weights(
        os.path.join(self.save_dir, 
          self.file_fmt.format(
            epoch = epoch, **logs)),
        overwrite = True)

  def on_batch_begin(self, batch, logs = None):
    pass

  def on_batch_end(self, batch, logs = None):
    pass


__all__ = ['create_base_network', 'dump_network_separately', 
  'load_network_separately', 'attach_classifier', 
  'SeparateSaveCallback']