# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras base class for convolution layers."""
import tensorflow
import tensorflow.compat.v2 as tf

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils
import sys

class ConvSymb(Layer):
    """Abstract N-D convolution layer (private, used as implementation base).

    This layer creates a convolution function that is computed with the layer
    input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Args:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution). Could be "None", eg in the case of
        depth wise convolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros
        evenly to the left/right or up/down of the input such that output has
        the same height/width dimension as the input. `"causal"` results in
        causal (dilated) convolutions, e.g. `output[t]` does not depend on
        `input[t+1:]`.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      groups: A positive integer specifying the number of groups in which the
        input is split along the channel axis. Each group is convolved
        separately with `filters / groups` filters. The output is the
        concatenation of all the `groups` results along the channel axis.
        Input channels and `filters` must both be divisible by `groups`.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied.
      use_bias: Boolean, whether the layer uses a bias.

      bias_initializer: An initializer for the bias vector. If None, the default
        initializer (zeros) will be used.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.

      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
    """

    def __init__(
        self,
        rank,
        filters,
        custom_function,

        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        **kwargs,
    ):
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs,
        )
        self.rank = rank

        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. "
                "Expected a strictly positive value. "
                f"Received filters={filters}."
            )
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, "kernel_size"
        )
        self.custom_function =custom_function
        self.strides = conv_utils.normalize_tuple(
            strides, rank, "strides", allow_zero=True
        )
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )

        self.activation = activations.get(activation)
        self.use_bias = use_bias


        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        self._validate_init()
        self._is_causal = self.padding == "causal"
        self._channels_first = self.data_format == "channels_first"
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2
        )

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the "
                "number of groups. Received: groups={}, filters={}".format(
                    self.groups, self.filters
                )
            )



        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0(s). Received: %s"
                % (self.strides,)
            )

        if self.padding == "causal":

            from keras.layers.convolutional.conv1d import Conv1D
            from keras.layers.convolutional.separable_conv1d import (
                SeparableConv1D,
            )

            if not isinstance(self, (Conv1D, SeparableConv1D)):
                raise ValueError(
                    "Causal padding is only supported for `Conv1D`"
                    "and `SeparableConv1D`."
                )
    def convolution_op(self, inputs, kernel):
        if self.padding == "causal":
            tf_padding = "VALID"  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        return tf.nn.convolution(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=tf_padding,
            dilations=list(self.dilation_rate),
            data_format=self._tf_data_format,
            name=self.__class__.__name__,
        )
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by "
                "the number of groups. Received groups={}, but the input "
                "has {} channels (full input shape is {}).".format(
                    self.groups, input_channel, input_shape
                )
            )
        kernel_shape = self.kernel_size + (
            input_channel // self.groups,
            self.filters,
        )

        # compute_output_shape contains some validation logic for the input
        # shape, and make sure the output shape has all positive dimensions.
        self.compute_output_shape(input_shape)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        self.built = True





    def call(self, inputs):
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))
        if inputs.shape[3]:
            outputs = self.custom_function(inputs[:,:,:,:3])

        else:
            outputs_1 = self.custom_function(inputs[:, :, :, :3])
            outputs_2 = self.custom_function(inputs[:, :, :, 3:6])
            outputs = tensorflow.concat(outputs_1,outputs_2,axis=4)

        outputs = self.convolution_op(outputs, self.kernel)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = tf.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return tf.nn.bias_add(
                            o, self.bias, data_format=self._tf_data_format
                        )

                    outputs = conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1
                    )
                else:
                    outputs = tf.nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format
                    )

        if not tf.executing_eagerly() and input_shape.rank:
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return spatial_input_shape

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        try:
            if self.data_format == "channels_last":

                return tf.TensorShape(
                    input_shape[:batch_rank]
                    + self._spatial_output_shape(input_shape[batch_rank:-1])
                    + [self.filters]
                )
            else:
                return tf.TensorShape(
                    input_shape[:batch_rank]
                    + [self.filters]
                    + self._spatial_output_shape(input_shape[batch_rank + 1 :])
                )

        except ValueError:
            raise ValueError(
                "One of the dimensions in the output is <= 0 "
                f"due to downsampling in {self.name}. Consider "
                "increasing the input size. "
                f"Received input shape {input_shape} which would produce "
                "output shape with a zero or negative value in a "
                "dimension."
            )

    def _recreate_conv_op(self, inputs):
        return False

    def get_config(self):
        config = {
            "filters": self.filters,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "bias_initializer": initializers.serialize(self.bias_initializer),

            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.input_shape - 1)
        if getattr(inputs.shape, "ndims", None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == "channels_last":
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == "channels_first":
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == "causal":
            op_padding = "valid"
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding