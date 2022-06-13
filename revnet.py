import layers
from layers import *


# Summarizing the Layers into a Revnet Block
class RevnetStep(Layer):
    def __init__(self, num_channels, type, **kwargs):
        super(RevnetStep, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.type = type

    def build(self, input_shape):
        self.actnorm = layers.ActNorm()
        self.convo_1x1 = layers.Convolution_1x1(self.type)
        self.affine_coupling = layers.AffineCoupling(self.num_channels)

    def call(self, inputs, log_det, forward=True):
        if forward:
            x = inputs
            x, log_det = self.actnorm(x, log_det, forward)
            x, log_det = self.convo_1x1(x, log_det, forward)
            x, log_det = self.affine_coupling(x, log_det, forward)
            return x, log_det
        else:
            z = inputs
            z, log_det = self.affine_coupling(z, log_det, forward)
            z, log_det = self.convo_1x1(z, log_det, forward)
            z, log_det = self.actnorm(z, log_det, forward)
            return z, log_det