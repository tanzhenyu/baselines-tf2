import os
import numpy as np
import tensorflow as tf
from collections import deque

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(input_shape, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.name_scope(scope):
        layer = tf.keras.layers.Dense(units=nh, kernel_initializer=ortho_init(init_scale),
                                      bias_initializer=tf.keras.initializers.Constant(init_bias))
        # layer = tf.keras.layers.Dense(units=nh, kernel_initializer=tf.keras.initializers.Constant(init_scale),
        #                               bias_initializer=tf.keras.initializers.Constant(init_bias))
        layer.build(input_shape)
    return layer

