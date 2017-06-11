import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from bhtsne import tsne

from data import *
from siam import contrastive_loss
from net import create_base_network

# workaround
from keras import losses
losses.contrastive_loss = contrastive_loss
# ---------- #

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)


def visualize_data(
  data, labels, model_path, max_iter = 1000):
  model = create_base_network(data.shape[1:])
  model.load_weights(model_path)
  results = model.predict(
    data, 96, verbose = 1).astype(np.float64)
  reduced_data = tsne(results, max_iter = max_iter)
  num_classes = labels.max() + 1
  for i in range(num_classes):
    plot_data = reduced_data[labels == i]
    plt.plot(
      plot_data[:, 0], 
      plot_data[:, 1], 
      color = np.random.rand(3),
      marker = 'o', linestyle = '', alpha = 0.7
    )
  plt.show()


__all__ = ['visualize_data']


if __name__ == '__main__':
  parser = ArgumentParser(description = 'Data visualizer')
  parser.add_argument('--data-path', '-d',
                      type = str,
                      help = 'Path to data file',
                      required = True)
  parser.add_argument('--model-path', '-m',
                      type = str,
                      help = 'Path to trained model',
                      required = True)
  parser.add_argument('--tsne-iter', '-i',
                      type = int,
                      help = 'Max number of tSNE iterations',
                      default = 1000)
  args = parser.parse_args()
  data_path = args.data_path
  model_path = args.model_path
  max_iter = args.tsne_iter
  data, labels = load_npz(data_path)
  visualize_data(data, labels, model_path, max_iter)
