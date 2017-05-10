import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
import os
import json


def create_base_network(input_dim):
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
  network_json = network.to_json()
  with open('%s.json' % name, 'w') as json_file:
      json_file.write(network_json)
  network.save_weights('%s.h5' % name)
  print('Saved %s to disk' % name)


def load_network(name = 'network'):
  json_file = open('%s.json' % name, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights('%s.h5' % name)
  print('Loaded %s from disk' % name)

  return loaded_model


def output_predictions(X_train, y_train, name):
  model = load_network(name)
  pred = model.predict(X_train)

  prefix = os.path.split(name)[1]
    
  np.savetxt('%s_predictions.csv' % prefix, pred, fmt = '%.9f', delimiter = ',')
  np.savetxt('%s_labels.csv' % prefix, y_train, fmt = '%d')


__all__ = ['create_base_network', 'dump_network', 'load_network', 'output_predictions']