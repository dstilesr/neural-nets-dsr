# Neural Nets

## Contents
* [About](#about)
* [Repository Contents](#repository-contents)
* [Branch 'Policies'](#branch-policies)
* [Current Issues](#current-issues)

## About
This repo contains some personal implementations of neural networks done for
practice implementing backprop and optimization from scratch (using only numpy).


## Branch 'Policies'
For development, push to the `develop` branch, then merge with master when ready.

## Repository Contents
The `neural_nets_dsr` package contains the following subpackages and modules:
- The `activations` package contains activation functions for network layers, along
  with an `ActivationFunc` class in the `base.py` module for creating new ones.

- The `cost_functions` package contains several cost functions that can be used
  to train networks, and it also contains a `base.py` module with a class that allows
  the creation of new cost functions.
 
- The `layers` package contains several layer implementations that can be used to
  construct networks.

- The `optim` package contains optimization algorithms for training.

- The `network.py` module contains the class that represents a network.

- Finally, the `utils.py` module is for miscellaneous utility functions and
  classes.

## Current Issues
- Convolution backprop not always working.
- Doubts about batchnorm derivative computation.
- Numerical stability.
- Strange error in setup when cythonizing: 
  `error: each element of 'ext_modules' option must be an Extension instance or 2-tuple`

[Back to top](#neural-nets)