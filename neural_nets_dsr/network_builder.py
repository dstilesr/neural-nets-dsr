from . import layers
from typing import Tuple, List
from .network import NeuralNet
from .constants import LAYER_KWARGS


def validate_input_dict(inputs: dict, key: str):
    """
    Checks that all the arguments required to initialize a layer are present
    in the given inputs.
    :param inputs: Dictionary of inputs.
    :param key: Key of the layer in the constants.LAYER_KWARGS dict.
    :return:
    """
    if not all([k in inputs.keys() for k in LAYER_KWARGS[key]]):
        raise KeyError(
            f"Not all required arguments for {key:s} layer were specified!"
        )


def build_conv_layer(
        input_shape: Tuple[int, ...],
        seed: int = 1,
        **kwargs) -> Tuple[layers.Convolution2D, Tuple]:
    """
    Create a convolutional layer.
    :param input_shape:
    :param seed:
    :param kwargs:
    :return: The layer and the shape of the outputs.
    """
    assert len(input_shape) == 4
    validate_input_dict(kwargs, "Conv2D")
    kwargs["prev_channels"] = input_shape[-1]
    layer = layers.Convolution2D.initialize(**kwargs, seed=seed)
    outshape = layer.output_shape(input_shape)
    return layer, outshape


def build_maxpool_layer(
        input_shape: Tuple[int, ...],
        seed: int = 1,
        **kwargs) -> Tuple[layers.MaxPool, Tuple]:
    """
    Build a max pooling layer.
    :param input_shape:
    :param seed:
    :param kwargs:
    :return:
    """
    assert len(input_shape) == 4
    validate_input_dict(kwargs, "MaxPool")
    lyr = layers.MaxPool.initialize(**kwargs)
    return lyr, lyr.get_output_shape(input_shape)


def build_avgpool_layer(
        input_shape: Tuple[int, ...],
        seed: int = 1,
        **kwargs) -> Tuple[layers.AvgPool, Tuple]:
    """
    Build an avg pooling layer.
    :param input_shape:
    :param seed:
    :param kwargs:
    :return:
    """
    assert len(input_shape) == 4
    validate_input_dict(kwargs, "AvgPool")
    lyr = layers.AvgPool.initialize(**kwargs)
    return lyr, lyr.get_output_shape(input_shape)


def build_flatten_layer(
        input_shape: Tuple[int, int],
        seed: int = 1,
        **kwargs) -> Tuple[layers.FlattenLayer, Tuple]:
    """

    :param input_shape:
    :param seed:
    :param kwargs:
    :return:
    """
    assert len(input_shape) == 4
    validate_input_dict(kwargs, "Flatten")
    layer = layers.FlattenLayer.initialize()
    outshape = (input_shape[1] * input_shape[2] * input_shape[3], 1)
    return layer, outshape


def build_dense_layer(
        input_shape: Tuple[int, int],
        seed: int = 1,
        **kwargs) -> Tuple[layers.DenseLayer, Tuple]:
    """
    Initialize a dense network layer.
    :param input_shape:
    :param seed:
    :param kwargs:
    :return: The layer and the shape of the outputs.
    """
    assert len(input_shape) == 2
    validate_input_dict(kwargs, "DenseLayer")

    kwargs["prev_layer_neurons"] = input_shape[0]
    layer = layers.DenseLayer.initialize(**kwargs, seed=seed)
    out_shape = (layer.biases.shape[0], 1)
    return layer, out_shape


def build_dropout_layer(
        input_shape: Tuple[int, ...],
        seed: int = 1,
        **kwargs) -> Tuple[layers.DropoutLayer, Tuple]:
    """
    Initialize a Dropout layer.
    :param input_shape:
    :param seed:
    :param kwargs:
    :return: The layer and the shape of the outputs.
    """
    validate_input_dict(kwargs, "Dropout")
    layer = layers.DropoutLayer.initialize(
        **kwargs,
        input_shape=list(input_shape),
        seed=seed
    )
    return layer, input_shape


def build_batchnorm_layer(
        input_shape: Tuple[int, ...],
        seed: int = 1,
        **kwargs) -> Tuple[layers.BatchNorm, Tuple]:
    """
    Initialize a BatchNorm layer.
    :param input_shape:
    :param seed:
    :param kwargs:
    :return: The layer and the shape of the outputs.
    """
    validate_input_dict(kwargs, "BatchNorm")
    layer = layers.BatchNorm.initialize(
        **kwargs,
        input_shape=input_shape,
        seed=seed
    )
    return layer, input_shape


FUNC_MAPPING = {
    "Conv2D": build_conv_layer,
    "MaxPool": build_maxpool_layer,
    "AvgPool": build_avgpool_layer,
    "Dropout": build_dropout_layer,
    "Flatten": build_flatten_layer,
    "BatchNorm": build_batchnorm_layer,
    "DenseLayer": build_dense_layer
}


def build_network(
        input_shape: Tuple[int, ...],
        layer_args: List[dict],
        init_seed: int = 1,
        verbose: bool = False) -> NeuralNet:
    """
    Builds a networks from a list of dictionaries containing the parameters to
    initialize each layer. Each dictionary mus contain a 'type' key specifying
    the kind of layer as well as other kwargs to pass to the layer's
    initialize() method.
    :param input_shape: Shape of inputs. Examples axis must be set to 1.
    :param layer_args: List of dictionaries to initialize layers.
    :param init_seed: Seed for RNG.
    :param verbose: Print basic layer info or not.
    :return: A NeuralNet instance.
    """
    shape = input_shape
    network_layers = []
    if verbose:
        print("INPUT SHAPE:", input_shape)

    for kwds in layer_args:
        lyr_type = kwds.pop("type")
        if lyr_type not in LAYER_KWARGS.keys():
            raise KeyError("Unknown layer type: %s" % lyr_type)

        layer, shape = FUNC_MAPPING[lyr_type](
            **kwds,
            seed=init_seed,
            input_shape=shape
        )
        if verbose:
            print(f"LAYER TYPE {lyr_type}, OUTPUT SHAPE: {shape}")

        network_layers.append(layer)
        init_seed += 1

    return NeuralNet(network_layers)
