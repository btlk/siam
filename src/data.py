import os, sys
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from argparse import ArgumentParser


def read_labeled_data(images_dir, labels_file):
  images_data = []
  labels_list = [int(x.strip()) 
    for x in open(labels_file, 'r').readlines()]

  images_list = sorted(os.listdir(images_dir))
  for im in images_list:
    with open(os.path.join(
      images_dir, im), 'rb') as img_stream:
      file_bytes = np.asarray(
        bytearray(img_stream.read()), dtype=np.uint8)
      img_data_ndarray = cv2.imdecode(
        file_bytes, cv2.IMREAD_UNCHANGED)
      images_data.append(img_data_ndarray)

  return np.asarray(images_data), \
    np.asarray(labels_list)


def read_labeled_data2(images_dir):
  dirs_list = os.listdir(images_dir)

  images_data = []
  labels_list = []

  for d in dirs_list:
    images_list = os.listdir(
      os.path.join(images_dir, d))
    for im in images_list:
      with open(os.path.join(
        images_dir, d, im), 'rb') as img_stream:
        file_bytes = np.asarray(
          bytearray(img_stream.read()), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(
          file_bytes, cv2.IMREAD_UNCHANGED)

        images_data.append(img_data_ndarray)
        labels_list.append(int(d))

  return np.asarray(images_data), \
    np.asarray(labels_list)


def read_data(images_dir):
  images_data = []

  images_list = sorted(os.listdir(images_dir))
  for im in images_list:
    with open(os.path.join(
      images_dir, im), 'rb') as img_stream:
      file_bytes = np.asarray(
        bytearray(img_stream.read()), dtype=np.uint8)
      img_data_ndarray = cv2.imdecode(
        file_bytes, cv2.IMREAD_UNCHANGED)
      images_data.append(img_data_ndarray)

  return np.asarray(images_data)


def convert_to_npz(images, labels, output_path):
  np.savez_compressed(output_path, 
    images = images, labels = labels)


def load_npz(path):
  images, labels = None, None
  with np.load(path) as npf:
    images = npf['images']
    labels = npf['labels']
  return images, labels


__all__ = ['read_data', 'read_labeled_data', 
  'read_labeled_data2', 'convert_to_npz', 'load_npz'] 


if __name__ == '__main__':
  parser = ArgumentParser(description = 'Data runner')
  parser.add_argument('--input-dir', '-i',
                      type = str,
                      nargs = '+',
                      help = 'Path to input directory',
                      required = True)
  parser.add_argument('--output-path', '-o',
                      type = str,
                      help = 'Path to output file',
                      required = True)
  args = parser.parse_args()
  input_path = args.input_dir
  output_path = args.output_path

  if len(input_path) == 1:
    images, labels = read_labeled_data2(input_path[0])
  else:
    images = []
    labels = []
    for i, p in enumerate(tqdm(input_path)):
      images.append(read_data(p))
      labels.append([i]*len(images[-1]))
    images = np.concatenate(images)
    labels = np.concatenate(labels)

  convert_to_npz(images, labels, output_path)



