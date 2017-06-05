import os, sys
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm
from argparse import ArgumentParser


def SlidingCrop(im, size, count):
  results = []
  h, w = im.shape[:2]
  step = int(size / 3)
  cur_y, cur_x = 0, 0
  end_y, end_x = h - size, w - size
  while (cur_y < end_y):
    while (cur_x < end_x):
      results.append(
        im[cur_y:cur_y + size, cur_x:cur_x + size])
      cur_x += step
    cur_x = 0
    cur_y += step
  if count != -1:
    results = random.sample(results, count)
  return results


def RandomCrop(im, size, count):
  results = []
  h, w = im.shape[:2]
  for i in range(count):
    x_ = np.random.randint(0, w - size)
    y_ = np.random.randint(0, h - size)
    results.append(im[y_:y_ + size, x_:x_ + size])
  return results


if __name__ == '__main__':
  parser = ArgumentParser(description = 'Data runner')
  parser.add_argument('--input-dir', '-i',
                      type = str,
                      help = 'Path to input directory',
                      required = True)
  parser.add_argument('--crop-size', '-s',
                      type = int,
                      help = 'Size of cropping square',
                      default = 48)
  parser.add_argument('--crop-count', '-c',
                      type = int,
                      help = 'Crops per image count\n' +
                        'Use \'-1\' with crop_mode ' +
                        '\'slid\' to keep all crops',
                      default = 25)
  parser.add_argument('--crop-mode', '-m',
                      type = str,
                      help = 'Crop mode (rand|slid)',
                      default = 'rand')
  parser.add_argument('--output-dir', '-o',
                      type = str,
                      help = 'Path to output directory',
                      required = True)
  args = parser.parse_args()
  input_path = args.input_dir
  output_path = args.output_dir
  crop_size = args.crop_size
  crop_count = args.crop_count
  crop_mode = args.crop_mode

  Crop = None
  if crop_mode == 'slid':
    Crop = SlidingCrop
  elif crop_mode == 'rand':
    Crop = RandomCrop
  else:
    raise RuntimeError(
      'Invalid crop mode \'%s\'' % crop_mode)

  images_list = os.listdir(input_path)
  if os.path.exists(output_path):
    shutil.rmtree(output_path)
  os.makedirs(output_path)

  for i in tqdm(images_list):
    im = cv2.imread(os.path.join(input_path, i))
    crops = Crop(im, crop_size, crop_count)
    for j, c in enumerate(crops):
      cv2.imwrite(os.path.join(output_path, 
        '%s_%d.png' % (i.split('.')[0], j)), c)