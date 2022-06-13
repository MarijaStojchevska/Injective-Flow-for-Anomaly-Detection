import scipy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


# Squeeze Layer
class Squeeze(Layer):
    def __init__(self,  **kwargs):
        super(Squeeze, self).__init__(**kwargs)

    def call(self, inputs, forward=True):
        B, H, W, C = inputs.shape
        if B is None:
            B = 1
        if forward:
            x = inputs
            x = tf.reshape(x, shape=[B, H // 2, 2, W // 2, 2, C])
            x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
            x = tf.reshape(x, shape=[B, H // 2, W // 2, C * 4])
            return x
        else:
            z = inputs
            z = tf.reshape(z, shape=[B, H, W, 2, 2, C // 4])
            z = tf.transpose(z, [0, 1, 3, 2, 4, 5])
            z = tf.reshape(z, shape=[B, H * 2, W * 2, C // 4])
            return z


# Activation Normalization Layer
class ActNorm(Layer):
    def __init__(self, **kwargs):
        super(ActNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        B, H, W, C = input_shape
        self.s = self.add_weight(shape=(1,1,1,C),  trainable=True, name="scale")
        self.b = self.add_weight(shape=(1,1,1,C), trainable=True, name="bias")
        self.initialized = self.add_weight(trainable=False, dtype=tf.bool, name="initialization")
        self.initialized.assign(False)

    @tf.function
    def call(self, inputs, log_det, forward=True):
        B, H, W, C = inputs.shape

        if not self.initialized:
            x = inputs
            assert (len(x.shape) == 4)
            # Calculate mean per channel
            mean = tf.math.reduce_mean(x, axis=[0,1,2], keepdims=True)
            # Calculate variance per channel
            var = tf.math.reduce_mean((x-mean)**2, axis=[0,1,2], keepdims=True)
            self.b.assign(-mean)
            self.s.assign(tf.math.log(1.0/(tf.math.sqrt(var) + 1e-6)))
            self.initialized.assign(True)

        if forward:
            x = inputs
            output = tf.math.multiply(x + self.b, tf.math.exp(self.s))
            log_det += H * W * tf.math.reduce_sum(self.s)
        else:
            z = inputs
            output = tf.math.multiply(tf.math.exp(-self.s), z) - self.b
            log_det -= H * W * tf.math.reduce_sum(self.s)

        return output, log_det


# 1x1 Convolution Layer
class Convolution_1x1(Layer):
    def __init__(self, type, **kwargs):
        super(Convolution_1x1, self).__init__()
        self.type = type
        self.gamma = kwargs.get('gamma', 0.0)

    def build(self, input_shape):
        _, H, W, C = input_shape

        if self.type == 'bijective':
            self.w = self.add_weight(shape=(1, 1, C, C), trainable=True, initializer=tf.keras.initializers.orthogonal, name="convo_weights")
        else:
            random_matrix_1 = np.random.randn(C // 2, C // 2).astype("float32")
            random_matrix_2 = np.random.randn(C // 2, C // 2).astype("float32")
            np_w1 = scipy.linalg.qr(random_matrix_1)[0].astype("float32")
            np_w2 = scipy.linalg.qr(random_matrix_2)[0].astype("float32")
            np_w = np.concatenate([np_w1, np_w2], axis=0) / (np.sqrt(2.0))
            self.w = tf.Variable(np_w, name='W', trainable=True)

            self.s = tf.linalg.svd(self.w, full_matrices=False, compute_uv=False)  # singular values of w
            self.log_s = tf.math.log(self.s + self.gamma ** 2 / (self.s + 1e-8))

    def call(self, inputs, log_det, forward=True):
        _, H, W, C = inputs.shape

        if forward:
            if self.type == 'bijective':
                x = inputs
                z = tf.nn.conv2d(x, self.w, strides=[1, 1, 1, 1], padding='SAME')
                determinant = tf.math.reduce_sum(tf.linalg.det(self.w))
                log_det += H * W * tf.math.log(tf.math.abs(determinant))
                return z, log_det
            else:
                x = inputs
                w = tf.reshape(self.w, [1, 1] + self.w.get_shape().as_list())
                z = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format="NHWC")

                log_det += tf.reduce_sum(self.log_s) * \
                           tf.cast(H * W, self.log_s.dtype)
                return z, log_det
        else:
            if self.type == 'bijective':
                z = inputs
                w_inverse = tf.linalg.inv(self.w)
                x = tf.nn.conv2d(z, w_inverse, strides=[1, 1, 1, 1], padding='SAME')
                determinant = tf.math.reduce_sum(tf.linalg.det(self.w))
                log_det -= H * W * tf.math.log(tf.math.abs(determinant))
                return x, log_det

            else:
                z = inputs
                prefactor = tf.matmul(self.w, self.w, transpose_a=True) + \
                            self.gamma ** 2 * tf.eye(tf.shape(self.w)[1])

                w_inv = tf.matmul(tf.linalg.inv(prefactor), self.w, transpose_b=True)
                conv_filter = w_inv
                conv_filter = tf.reshape(conv_filter, [1, 1] + conv_filter.get_shape().as_list())
                x = tf.nn.conv2d(z, conv_filter, [1, 1, 1, 1], "SAME", data_format="NHWC")

                log_det -= tf.reduce_sum(self.log_s) * \
                           tf.cast(H * W, self.log_s.dtype)
                return x, log_det


# Affine Coupling Layer
class AffineCoupling(Layer):
    def __init__(self, num_channels, **kwargs):
        super(AffineCoupling, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.net = tf.keras.Sequential()

    def build(self, input_shape):
        B,H,W,C = input_shape
        self.net.add(tf.keras.layers.Conv2D(self.num_channels, kernel_size=3, activation='relu', strides=(1,1), padding='same'))
        self.net.add(tf.keras.layers.Conv2D(self.num_channels, kernel_size=1, activation='relu', strides=(1,1), padding='same'))
        self.net.add(zeroInitialization(C))

    def call(self, inputs, log_det, forward=True):
        x1, x2 = tf.split(inputs, num_or_size_splits=2, axis=3)
        output = self.net(x1)
        shift = output[:, :, :, 0::2]
        scale = tf.nn.sigmoid(output[:, :, :, 1::2] + 2.0)
        scale = 0.01 + (1-0.01)*scale

        z1 = x1

        if forward:
            z2 = tf.math.multiply((x2 + shift), scale)
            log_det_term = tf.math.log(tf.math.abs(scale))
            log_det += tf.math.reduce_mean(tf.math.reduce_sum(log_det_term, axis=[1, 2, 3]))
            z = tf.concat([z1, z2], axis=3)
            return z, log_det

        else:
            z2 = tf.math.divide(x2, scale) - shift
            log_det_term = tf.math.log(tf.math.abs(scale))
            log_det -= tf.math.reduce_mean(tf.math.reduce_sum(log_det_term, axis=[1, 2, 3]))
            x = tf.concat([z1, z2], axis=3)
            return x, log_det


# Split Layer
class Split(Layer):
    def __init__(self):
        super(Split, self).__init__()

    def build(self, input_shape):
        B, H, W, C = input_shape
        self.gaussianize = Gaussianize(num_filters=C)

    def call(self, inputs, forward=True):

        if forward:
            x = inputs
            x1, x2 = tf.split(x, num_or_size_splits=2, axis=3)
            z_i, parameters = self.gaussianize([x1, x2], forward)
            mu, sigma = parameters
            log_pz = log_probability(z_i, mu, sigma)
            log_pz = tf.math.reduce_mean((tf.math.reduce_sum(log_pz, axis=[1, 2, 3])), axis=0)
            return x1, z_i, log_pz

        else:
            x1, z_i = inputs[0], inputs[1]
            x2 = self.gaussianize(inputs, forward)
            x = tf.concat([x1, x2], axis=3)
            return x


class Gaussianize(Layer):
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters

    def build(self, input_shape):
        self.net = tf.keras.layers.Conv2D(self.num_filters, kernel_size=3, kernel_initializer="zeros", bias_initializer="zeros", strides=(1, 1), padding="same", activation=None)
        self.log_scale_factor = self.add_weight(shape=(1, 1, 1, self.num_filters), trainable=True, initializer="zeros", name="log_scale")

    def call(self, inputs, forward=True):
        if forward:
            x1 = inputs[0]
            x2 = inputs[1]
            h = self.net(x1) * tf.math.exp(self.log_scale_factor * 2)
            mu, sigma = tf.split(h, num_or_size_splits=2, axis=3)
            z_i = x2
            parameters = [mu, sigma]
            return z_i, parameters

        else:
            x1 = inputs[0]
            z_i = inputs[1]
            h = self.net(x1) * tf.math.exp(self.log_scale_factor * 3)
            mu, sigma = tf.split(h, num_or_size_splits=2, axis=3)
            x2 = mu + z_i * tf.math.exp(sigma)
            return x2


class zeroInitialization(Layer):
    def __init__(self, num_filters, **kwargs):
        super(zeroInitialization, self).__init__(**kwargs)
        self.num_filters = num_filters

    def build(self, input_shape):
        self.zeroLayer = tf.keras.layers.Conv2D(self.num_filters, kernel_size=3, kernel_initializer="zeros", bias_initializer="zeros", strides=(1, 1), padding="same", activation=None)
        self.scale = self.add_weight(shape=(1, 1, 1, self.num_filters), trainable=True, initializer="zeros", name="zeroInitializer_scale")

    def call(self, inputs):
        output = self.zeroLayer(inputs)
        output = output * tf.math.exp(self.scale * 3.0)
        return output

def log_probability(z, mu, sigma):
    return -0.5 * tf.math.log(2 * np.pi) - sigma - 0.5 * (z - mu) ** 2 / tf.math.exp(2 * sigma)