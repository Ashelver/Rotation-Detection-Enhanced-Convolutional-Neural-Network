import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence
from collections import defaultdict
from Image_oscillation import BatchWise_Oscillate
from Polluted_Images_Generation import CRRNWEP


POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)



class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        conv_channels = [in_channels] + self.channels # Add the input channal into the channal

        for i in range(len(self.channels)):
            # Create convolution layer
            conv_layer = nn.Conv2d(conv_channels[i], conv_channels[i+1], **self.conv_params)
            layers.append(conv_layer)

            # Apply activation
            activation_fn = ACTIVATIONS.get(self.activation_type)(**self.activation_params)
            layers.append(activation_fn)

            # Pooling every `pool_every` layers
            if (i + 1) % self.pool_every == 0:
                pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling_layer)

        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # Set up dummy input to pass through the layers
            dummy_input = torch.rand(1, *self.in_size)  # A single sample, with the correct shape (C, H, W)
            dummy_output = self.feature_extractor(dummy_input)  # Pass it through the feature extractor
            extracted_features = int(torch.prod(torch.tensor(dummy_output.shape[1:]))) # (channels * height * width)
        finally:
            torch.set_rng_state(rng_state)
        return extracted_features

    def _make_mlp(self):
        # Create the MLP part of the model: (FC -> ACT)*M -> Linear
        mlp_layers = []
        in_features = self._n_features()

        activation_fn = ACTIVATIONS.get(self.activation_type)(**self.activation_params)
        mlp: nn.Module = None
        # Create hidden layers
        for dim in self.hidden_dims:
            mlp_layers.append(nn.Linear(in_features, dim))
            mlp_layers.append(activation_fn)  # Activation function
            in_features = dim

        # Create output layer
        mlp_layers.append(nn.Linear(in_features, self.out_classes))

        mlp = nn.Sequential(*mlp_layers)
        return mlp

    def forward(self, x: Tensor):
        # Implement the forward pass.
        # Extract features from the input using the feature extractor
        features = self.feature_extractor(x)

        # Flatten the features to feed them to the MLP
        features = features.view(features.size(0), -1)

        out: Tensor = None
        # Pass through the MLP classifier
        out = self.mlp(features)
        return out
    



class Oscillated_Block(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        
        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.bypath_one, self.bypath_two = None, None, None

        # Main path
        layers_main = []
        layers_shortcut_one = []
        layers_shortcut_two = []
        current_channels = in_channels
        num_conv = len(channels)
        activation = ACTIVATIONS[activation_type](**activation_params)
        for i in range(num_conv - 1):
            # Add convolution
            layers_main.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=channels[i],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2,
                    bias=True,  # Bias is enabled for main path
                )
            )
            # Add optional dropout
            if dropout > 0.0:
                layers_main.append(nn.Dropout2d(dropout))
            # Add optional batch normalization
            if batchnorm:
                layers_main.append(nn.BatchNorm2d(channels[i]))
            # Add activation function
            layers_main.append(activation)

            current_channels = channels[i]

        # Final convolution layer in the main path
        layers_main.append(
            nn.Conv2d(
                in_channels=current_channels,
                out_channels=channels[-1],
                kernel_size=kernel_sizes[-1],
                stride=2,
                bias=True,  # Final conv in main path uses bias
            )
        )
        self.main_path = nn.Sequential(*layers_main)

        # Shortcut path one
        layers_shortcut_one.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=2,
                stride=2,
                padding=2 // 2,
                bias=False,  # No bias for shortcut path  
            ),
        )
        # Add optional dropout
        if dropout > 0.0:
            layers_shortcut_one.append(nn.Dropout2d(dropout))
        # Add optional batch normalization
        if batchnorm:
            layers_shortcut_one.append(nn.BatchNorm2d(channels[i]))
        # Add activation function
        layers_shortcut_one.append(activation)

        layers_shortcut_one.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=channels[-1],
                kernel_size=2,
                bias=False,  # No bias for shortcut path  
            ),
        )


        self.bypath_one = nn.Sequential(*layers_shortcut_one)



        # Shortcut path two
        layers_shortcut_two.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=2,
                stride=2,
                padding=2 // 2,
                bias=False,  # No bias for shortcut path  
            ),
        )
        # Add optional dropout
        if dropout > 0.0:
            layers_shortcut_two.append(nn.Dropout2d(dropout))
        # Add optional batch normalization
        if batchnorm:
            layers_shortcut_two.append(nn.BatchNorm2d(channels[i]))
        # Add activation function
        layers_shortcut_two.append(activation)

        layers_shortcut_two.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=channels[-1],
                kernel_size=2,
                bias=False,  # No bias for shortcut path  
            ),
        )


        self.bypath_two = nn.Sequential(*layers_shortcut_two)


    def forward(self, x: Tensor):
        # Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        main_output = self.main_path(x)
        one , two = BatchWise_Oscillate(x).get_result()
        shortcut_output_one = self.bypath_one(one)
        shortcut_output_two = self.bypath_two(two)
        out = torch.cat([main_output, shortcut_output_one, shortcut_output_two], dim=1)  # [64, 48, 28, 28]
        return out

class RDENet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )


    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        conv_channels = [48] + self.channels # Add the input channal into the channal

        layers.append(
            Oscillated_Block(
                in_channels=in_channels, 
                channels=[16,16], 
                kernel_sizes=[2]*2,
                batchnorm=self.batchnorm,
                dropout=self.dropout, 
                activation_type=self.activation_type, 
                activation_params=self.activation_params
            )
        )

        pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
        layers.append(pooling_layer)

        for i in range(len(self.channels)):
            # Create convolution layer
            conv_layer = nn.Conv2d(conv_channels[i], conv_channels[i+1], **self.conv_params)
            layers.append(conv_layer)

            # Apply activation
            activation_fn = ACTIVATIONS.get(self.activation_type)(**self.activation_params)
            layers.append(activation_fn)

            # Pooling every `pool_every` layers
            if (i + 1) % self.pool_every == 0:
                pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling_layer)

        seq = nn.Sequential(*layers)
        return seq


