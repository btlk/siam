# siam

## brief

This repo contains suplementary code for my bachelor diploma @ NUST MISIS.

## in particular

`src` contains 4 files:
- `data.py` - useful utilities for data preparation
- `net.py` - network ops
- `siam.py` - training siamese net
- 'classifier.py' - training classifier on top of siamese net
- `visual.py` - visualizing data in prediction space using bhtsne
- 'crop.py' - app for cropping regions from arbitrary images

## credits

```
The siamese part is based on Keras siamese MNIST example:
https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py

Visualization is done using @dominiek 's `bhtsne` package
https://github.com/dominiek/python-bhtsne
```

## future plans

```
It's very likely (as I'm willing to grad the next month) that there will be frequent commits here.
Feel free to use my workarounds. -btlk
```
