# Neural Nets

## Contents
* [About](#about)
* [Repository Contents](#repository-contents)
* [Organization](#organization)
* [Branch 'Policies'](#branch-policies)
* [Current Issues](#current-issues)

## About
This repo contains some personal implementations of neural networks done for
practice implementing backprop and optimization from scratch (using only numpy).
Currently all networks are sequential.

To install as python package, run:
```bash
python setup.py install
```

To compile Cython modules for testing run:
```bash
python setup.py build_ext --inplace
```

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

## Organization
- Every activation and cost function should 'know' how to compute its own gradient.
- Each layer should know how to forward and back propagate through itself.
- Every optimizer should know how to perform updates on weights and biases.

## Current Issues
- Convolution backprop not always working.
- Doubts about batchnorm derivative computation.
- Numerical stability.

[Back to top](#neural-nets)