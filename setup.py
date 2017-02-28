from setuptools import setup, find_packages

import os

reqs = [x.strip() for x in open('REQUIREMENTS.txt', 'r').readlines()]

scripts = [os.path.join(d, x)
           for d, dirs, files in os.walk('bin')
           for x in files if x.endswith('.py')]
scripts_dir = os.path.expanduser('~/siam/bin')
if not os.path.exists(scripts_dir):
    os.makedirs(scripts_dir)

setup(
  name='siam',
  version='0.1',
  packages=['siam'],
  package_dir={'siam': 'src'},
  data_files=[(scripts_dir, scripts)],

  install_requires=reqs,

  author='Alexander \'btlk\' Serikov',
  author_email='ivanmitrafanych@gmail.com',
  description='Tensorflow implementation of siamese image comparing net',
  download_url='https://github.com/btlk/siam/',
  license='MIT',
  keywords=['tensorflow', 'neural', 'network', 'convolution', 'siamese', 'image']
)