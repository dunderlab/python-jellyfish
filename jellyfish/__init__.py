"""
======================================================================
EEGNet: A Deep Learning Architecture for EEG Signal Classification
======================================================================

EEGNet is designed as a convolutional neural network for the task of electroencephalogram (EEG)
signal classification. It provides a robust framework for applications ranging from brain-computer
interfaces to medical diagnosis. With customizable hyperparameters, EEGNet can be tailored to a
variety of EEG decoding tasks.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    Input,
    SeparableConv2D,
    SpatialDropout2D,
)
from tensorflow.keras.constraints import max_norm


class EEGNet:
    """EEGNet architecture for EEG signal classification."""

    def __init__(
        self,
        nb_classes: int,
        Chans: int = 64,
        Samples: int = 128,
        dropoutRate: float = 0.5,
        kernLength: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        norm_rate: float = 0.25,
        dropoutType: str = 'Dropout',
    ) -> None:
        """Initialize EEGNet with given parameters.

        Args:
            nb_classes (int): Number of classification labels.
            Chans (int, optional): Number of EEG channels. Defaults to 64.
            Samples (int, optional): Number of time samples in the EEG data. Defaults to 128.
            dropoutRate (float, optional): Dropout rate for regularization. Defaults to 0.5.
            kernLength (int, optional): Kernel length for convolutional layers. Defaults to 64.
            F1 (int, optional): Number of temporal filters. Defaults to 8.
            D (int, optional): Number of spatial filters per temporal filter. Defaults to 2.
            F2 (int, optional): Number of pointwise filters. Defaults to 16.
            norm_rate (float, optional): Max norm rate for kernel constraint. Defaults to 0.25.
            dropoutType (str, optional): The type of dropout layer. Defaults to 'Dropout'.

        Raises:
            ValueError: If an invalid dropout type is specified.
        """
        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError(
                'dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

        input1 = Input(shape=(1, Chans, Samples))

        # Block 1: Temporal Convolution
        # This block applies a convolution across the time dimension with F1 filters,
        # then a depthwise convolution that separately learns spatial filters for each
        # temporal filter, followed by batch normalization, an activation function,
        # average pooling to reduce dimensionality and a dropout layer for regularization.
        block1 = Conv2D(F1, (1, kernLength), padding='same',
                        input_shape=(1, Chans, Samples), use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                                 depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutType(dropoutRate)(block1)

        # Block 2: Separable Convolution
        # This block uses separable convolution with F2 pointwise filters, which
        # is effective in parameter reduction and computational cost. It's followed by
        # batch normalization, an activation function, average pooling, and a dropout layer.
        block2 = SeparableConv2D(
            F2, (1, 16), use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutType(dropoutRate)(block2)

        # Flatten and Dense Layers: Classification
        # This part flattens the output of the previous block to create a single
        # long feature vector, then applies a dense layer with a number of units equal
        # to the number of classes intended for classification, constrained by a max norm.
        # It concludes with a softmax activation function that outputs probability distributions
        # over the classes.
        flatten = Flatten(name='flatten')(block2)
        dense = Dense(nb_classes, name='dense',
                      kernel_constraint=max_norm(norm_rate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)

        self.model = Model(inputs=input1, outputs=softmax)
