import numpy as np
import random
from argparse import ArgumentParser
from keras.models import Model
from keras.utils import plot_model, to_categorical
from keras import backend as K
import tensorflow as tf
import os, shutil

from data import load_npz, convert_to_npz
from net import attach_classifier, create_base_network
from siam import contrastive_loss

from bhtsne import tsne

# workaround
from keras import losses
losses.contrastive_loss = contrastive_loss
# ---------- #


batch_size = 128

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)


def GetInliers(preds, mean, std):
  pts = (preds - mean) / std
  
  return np.power(pts, 2).sum(axis = 1) <= 1.0


__all__ = ['GetInliers']


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
  parser.add_argument('--output-dir', '-o',
                      type = str,
                      help = 'Path to output directory',
                      required = True)
  parser.add_argument('--max-iter', '-t',
                      type = int,
                      help = 'TSNE max iterations',
                      default = 1000)
  args = parser.parse_args()
  input_path = args.input_data
  model_path = args.model
  output_dir = args.output_dir
  max_iter = args.max_iter
  X_input, y_input = load_npz(input_path)
  X_input_train = X_input.astype(np.float32) / 255.0
  
  model = create_base_network(X_input_train.shape[1:])
  model.load_weights(model_path)

  preds = model.predict(X_input_train, verbose = 1).astype(np.float64)
  red_preds = tsne(preds, max_iter = max_iter)
  num_classes = y_input.max() + 1
  generations = [[] for j in range(3)]
  for c in range(num_classes):
    cur_preds = red_preds[y_input == c]
    cur_mean = cur_preds.mean(0)
    cur_std = cur_preds.std(0)
    
    X_input_c = X_input.copy()[y_input == c]
    print ('Original shape: %s' % cur_preds.shape)
    for j in range(3):
      inliers = GetInliers(cur_preds, cur_mean, cur_std)
      generations[j].append(X_input_c[inliers])
      cur_preds = cur_preds[inliers]
      print ('Generation %d shape: %s' % (j, cur_preds.shape))
      X_input_c = X_input_c[inliers]
      cur_mean = cur_preds.mean(0)
      cur_std = cur_preds.std(0)

  base_name = os.path.split(input_path)[1].split('.')[0]
  for i, g in enumerate(generations):
    convert_to_npz(
      np.concatenate(g), 
      np.concatenate([[0]*len(g[0]), [1]*len(g[1])]),
      os.path.join(output_dir, '%s_gen%d.npz' % (base_name, i))
    )


  