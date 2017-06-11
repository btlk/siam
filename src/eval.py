import numpy as np
import random
from argparse import ArgumentParser
from keras.models import Model
from keras.utils import plot_model, to_categorical
from keras import backend as K
import tensorflow as tf
import os, shutil

from data import load_npz
from net import attach_classifier, create_base_network
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

if __name__ == '__main__':
  parser = ArgumentParser(description = 'Data runner')
  parser.add_argument('--input-data', '-i',
                      type = str,
                      help = 'Path to input data',
                      required = True)
  parser.add_argument('--model', '-m',
                      type = str,
                      help = 'Path to trained classifier',
                      required = True)
  args = parser.parse_args()
  input_path = args.input_data
  model_path = args.model
  X_input, y_input = load_npz(input_path)
  X_input = X_input.astype(np.float32) / 255.0
  
  model = create_base_network(X_input.shape[1:])
  model = attach_classifier(model, 2)
  model.load_weights(model_path)

  preds = model.predict(X_input, verbose = 1)
  correct = np.equal(np.argmax(preds, 1), y_input).mean()
  correct_signs = np.equal(np.argmax(preds[:1400], 1), 0).mean()
  print ('Total: %d, correct: %.2f%%, correct_signs: %.2f%%'
    % (len(y_input), correct * 100, correct_signs * 100))