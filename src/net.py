import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import (Dense, Dropout, Input, 
  Conv2D, MaxPooling2D, Flatten)
import tensorflow as tf
import os
import json


def create_base_network(input_dim):
  seq = Sequential()
  seq.add(Conv2D(
    filters = 64, 
    kernel_size = (3, 3), 
    input_shape = input_dim,
    activation = 'relu'))
  seq.add(Conv2D(
    filters = 32, 
    kernel_size = (3, 3), 
    input_shape = input_dim, 
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
  seq.add(Flatten())
  seq.add(Dense(
    512, 
    activation='relu'))
  seq.add(Dropout(0.5))
  seq.add(Dense(
    128, 
    activation='relu'))

  return seq


def attach_classifier(model, output_pins):
  for l in model.layers:
    l.trainable = False
  # model.add(Dense(
  #   16,
  #   activation='relu',
  #   name = 'dense_clas_1'
  # ))
  model.add(Dense(
    output_pins,
    activation='softmax',
    name = 'dense_clas_2'
  ))
  return model


def dump_network_separately(network, name = 'network'):
  network_json = network.to_json()
  with open('%s.json' % name, 'w') as json_file:
      json_file.write(network_json)
  network.save_weights('%s.h5' % name)
  print('Saved %s to disk' % name)


def load_network_separately(name = 'network'):
  json_file = open('%s.json' % name, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights('%s.h5' % name)
  print('Loaded %s from disk' % name)

  return loaded_model


__all__ = ['create_base_network', 'dump_network_separately', 
  'load_network_separately', 'attach_classifier']