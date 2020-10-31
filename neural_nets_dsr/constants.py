

"""Parameters required to initialize each kind of layer."""
LAYER_KWARGS = {
    "Conv2D": [
        "num_filters",
        "filter_size",
        "padding",
        "activation"
    ],
    "MaxPool": ["filter_size"],
    "AvgPool": ["filter_size"],
    "Dropout": ["dropout_rate"],
    "Flatten": [],
    "BatchNorm": [],
    "DenseLayer": ["num_neurons", "activation"]
}
