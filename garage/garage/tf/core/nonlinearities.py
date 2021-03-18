import numpy as np
import tensorflow as tf


class ScaledShiftedSigmoid(object):
    def __init__(self, scale_in=1, scale_out=1, bias=0):
        self.scale_in = scale_in
        self.scale_out = scale_out
        self.bias = bias

    def __call__(self, x):
        return tf.nn.sigmoid(
            x * self.scale_in) * self.scale_out + self.bias
