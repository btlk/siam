import os, sys
import tarfile
import urllib.request
import re
import pickle

import tensorflow as tf
import numpy as np

def maybe_download_and_extract(dest_directory, data_url):
  '''Download and extract the tarball from Alex's website.'''
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  extraction_needed = False
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
      float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.') 
    extraction_needed = True
  with tarfile.open(filepath, 'r:gz') as t:
    dataset_dir = os.path.join(dest_directory, t.getmembers()[0].name)
    if (extraction_needed):
      t.extractall(dest_directory)
  
  return dataset_dir

#hyper parameters
n_classes=10
batch_size=256
image_width=32
image_height=32
image_depth=3
learning_rate=0.01
n_epochs=20
n_validate_samples=2000
n_test_samples=5

#regular expression that matches a datafile
r_data_file = re.compile('^data_batch_\d+')

def unpickle(relpath):  
  with open(relpath, 'rb') as fp:
    d = pickle.load(fp, encoding='latin1')
  return d

def prepare_input(data=None, labels=None):
  global image_height, image_width, image_depth
  assert(data.shape[1] == image_height * image_width * image_depth)
  assert(data.shape[0] == labels.shape[0])
  #do mean normaization across all samples
  mu = np.mean(data, axis = 0)
  mu = mu.reshape(1, -1)
  sigma = np.std(data, axis = 0)
  sigma = sigma.reshape(1, -1)
  data = data - mu
  data = data / sigma
  is_nan = np.isnan(data)
  is_inf = np.isinf(data)
  if np.any(is_nan) or np.any(is_inf):
    print('data is not well-formed : is_nan %s, is_inf: %s' % (np.any(is_nan), np.any(is_inf)))
  #data is transformed from (no_of_samples, 3072) to (no_of_samples , image_height, image_width, image_depth)
  #make sure the type of the data is no.float32
  data = data.reshape([-1, image_depth, image_height, image_width])
  data = data.transpose([0, 2, 3, 1])
  data = data.astype(np.float32)
  return data, labels

  #return transform_input(data=data, labels=labels, h=image_height, w=image_width, d=image_depth)

def convert_to_rgb_img_data(index = -1, data = None):
  assert(index < data.shape[0])
  image_holder = np.zeros(shape = [data.shape[1],data.shape[2], data.shape[3]], dtype = np.float32)
  image_holder[:, :, :] = data[index, :, :, :]
  plt.imshow(image_holder)
  
def load_and_preprocess_input(dataset_dir = None):
  assert(os.path.isdir(dataset_dir))

  #training and validate datasets as numpy n-d arrays, 
  #apropriate portions of which are ready to be fed to the placeholder variables
  train_all = {'data': [], 'labels': []}
  validate_all = {'data': [], 'labels': []}
  test_all = {'data': {}, 'labels': []}
  label_names_for_validation_and_test = None

  trn_all_data = []
  trn_all_labels = []
  vldte_all_data = []
  vldte_all_labels = []
  tst_all_data = []
  tst_all_labels = []
  #for loading train dataset, iterate through the directory to get matchig data file
  for root, dirs, files in os.walk(dataset_dir):
    for f in files:
      m = r_data_file.match(f)
      if m:
        relpath = os.path.join(root, f)
        d = unpickle(os.path.join(root, f))
        trn_all_data.append(d['data'])
        trn_all_labels.append(d['labels'])
  #concatenate all the  data in various files into one ndarray of shape
  #data.shape == (no_of_samples, 3072), where 3072=image_depth x image_height x image_width
  #labels.shape== (no_of_samples)
  trn_all_data = np.concatenate(trn_all_data).astype(np.float32)
  trn_all_labels = np.concatenate(trn_all_labels).astype(np.int32)
  
  #load the only test data set for validation and testing
  #use only the first n_validate_samples samples for validating
  test_temp = unpickle(os.path.join(dataset_dir, 'test_batch'))
  vldte_all_data = test_temp['data'][0:(n_validate_samples + n_test_samples), :]
  vldte_all_labels = test_temp['labels'][0:(n_validate_samples + n_test_samples)]
  vldte_all_data = np.concatenate([vldte_all_data]).astype(np.float32)
  vldte_all_labels = np.concatenate([vldte_all_labels]).astype(np.int32)
   #transform the test images in the same manner as the train images                                     
  train_all['data'], train_all['labels'] = prepare_input(data = trn_all_data, labels = trn_all_labels)
  validate_and_test_data, validate_and_test_labels = prepare_input(data = vldte_all_data, labels = vldte_all_labels)
  
  validate_all['data'] = validate_and_test_data[0:n_validate_samples, :, :, :]
  validate_all['labels'] = validate_and_test_labels[0:n_validate_samples]
  test_all['data'] = validate_and_test_data[n_validate_samples:(n_validate_samples + n_test_samples), :, :, :]
  test_all['labels'] = validate_and_test_labels[n_validate_samples:(n_validate_samples + n_test_samples)]
  
  #load all label-names
  label_names_for_validation_and_test = unpickle(os.path.join(dataset_dir, 'batches.meta'))['label_names'] 

  return train_all, validate_all, test_all, label_names_for_validation_and_test