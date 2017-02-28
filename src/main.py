import cifar10 as cifar

import os, sys

import tensorflow as tf
import numpy as np

def main():
  # obtain cifar10 if needed
  cifar_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  dataset_dir = cifar.maybe_download_and_extract(os.path.expanduser('~/siam/data'), cifar_url)
  print('%s batches reside in:' % (os.path.split(cifar_url)[1]))
  print(dataset_dir)

  trn, val, tst, val_tst_labels = cifar.load_and_preprocess_input(dataset_dir)
  print (len(trn), len(val), len(tst), len(val_tst_labels))

if __name__ == '__main__':
  main()