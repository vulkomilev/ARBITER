import tensorflow as tf
import numpy as np


class LocalInitializer(object):
    """Initializer base class: all Keras initializers inherit from this class.

    Initializers should implement a `__call__` method with the following
    signature:

    ```python
    def __call__(self, shape, dtype=None, **kwargs):
      # returns a tensor of shape `shape` and dtype `dtype`
      # containing values drawn from a distribution of your choice.
    ```

    Optionally, you an also implement the method `get_config` and the class
    method `from_config` in order to support serialization -- just like with
    any Keras object.

    Here's a simple example: a random normal initializer.

    ```python
    import tensorflow as tf

    class ExampleRandomNormal(tf.keras.initializers.Initializer):

      def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

      def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(
            shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

      def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}
    ```

    Note that we don't have to implement `from_config` in the example above since
    the constructor arguments of the class the keys in the config returned by
    `get_config` are the same. In this case, the default `from_config`
    works fine.
    """

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor.
          **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.

        Returns:
          A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.

        Example:

        ```python
        initializer = RandomUniform(-1, 1)
        config = initializer.get_config()
        initializer = RandomUniform.from_config(config)
        ```

        Args:
          config: A Python dictionary, the output of `get_config`.

        Returns:
          A `tf.keras.initializers.Initializer` instance.
        """
        config.pop('dtype', None)
        return cls(**config)


class LocalConstant(LocalInitializer):
    """Initializer that generates tensors with constant values.

    Also available via the shortcut function `tf.keras.initializers.constant`.

    Only scalar values are allowed.
    The constant value provided must be convertible to the dtype requested
    when calling the initializer.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Constant(3.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Constant(3.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      value: A Python scalar.
    """

    def __init__(self, array=0):
        self.value = array

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to `self.value`.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. If not specified,
           `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`).
          **kwargs: Additional keyword arguments.
        """
        del kwargs
        return tf.convert_to_tensor(self.value)

    def get_config(self):
        return {'value': self.value}


class AutoEncoderCustomKernel(object):

    def __init__(self):
        self.init_neural_network()

    def init_neural_network(self, width, height, depth):
        encoder_input = tf.keras.Input((width, height, depth))
        encoder = tf.keras.layers.Conv2D((3, 3), (3, 3))(encoder_input)
        encoder = tf.keras.layers.AveragePooling2D()(encoder)
        encoder = tf.keras.layers.Conv2D((3, 3), (3, 3))(encoder_input)
        encoder = tf.keras.layers.AveragePooling2D()(encoder)
        encoder = tf.keras.layers.Dense(256)(encoder)

        decoder_inputs = tf.keras.Input((128))
        decoder = tf.keras.layers.Dense(128)(decoder_inputs)
        decoder = tf.keras.layers.Dense(128)(decoder)
        decoder = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(3, 3), kernel_initializer= \
            LocalConstant([[1.0, 0.0], [0.0, 1.0]]))(decoder_inputs)
        decoder = tf.keras.layers.Conv2DTranspose(filers=64, kernel_size=(3, 3), strides=(3, 3))(decoder)
