from setuptools import setup, find_packages

reqs = [x.strip() for x in open('REQUIREMENTS.txt', 'r').readlines()]

setup(
  name="siam",
  version="0.1",
  packages=find_packages('src'),
  package_dir={'': 'src'},
  scripts=[],

  install_requires=reqs,

  author="Alexander 'btlk' Serikov",
  author_email="ivanmitrafanych@gmail.com",
  description="Tensorflow implementation of siamese image comparing net",
  license="MIT",
  keywords="tensorflow neural network convolution siamese image"
)