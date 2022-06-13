from revnet import *
from tensorflow import keras
import tensorflow_probability as tfp


# INJECTIVE FLOW
class InjectiveFlow(keras.Model):
    def __init__(self, img_shape, num_channels, levels):
         super(InjectiveFlow, self).__init__()

         _, self.H, self.W, self.C = img_shape
         self.latent_shape = img_shape
         self.dimension = self.H * self.W * self.C
         self.num_channels = num_channels
         self.levels = levels
         self.squeeze_block = []
         self.bijective_block = []
         self.injective_block = []

         for l in range(levels):
             self.squeeze_block.append(Squeeze())
             self.bijective_block.append(RevnetStep(num_channels, type='bijective'))
             self.injective_block.append(RevnetStep(num_channels, type='injective'))

    def latent(self, latent_shape):
        self.latent_shape = latent_shape

    def call(self, inputs, forward=True, reconstruction=False):

        if forward:
            x = inputs
            log_det = 0.0
            for l in range(self.levels):
               # print("----------INJECTIVE LEVEL----------:", l)
               # print("X shape:", x.shape)
                x = self.squeeze_block[l](x, forward=forward)
               # print("Squeeze:", x.shape)
                x, log_det = self.bijective_block[l](x, log_det, forward=forward)
               # print("Bijective block:", x.shape)
                x, log_det = self.injective_block[l](x, log_det, forward=forward)
               # print("Injective block:", x.shape)
                z=x
                latent_dimension = np.prod(z.shape[1:])
               # print("Latent Dimension: ", latent_dimension)
            return z

        else:
            z = inputs
            B, C, H, W = self.latent_shape
            if reconstruction:
                z = tf.reshape(z, shape=(1, C, H, W))
            else:
                z = tf.reshape(z, shape=(B, C, H, W))
            log_det = 0.0
            for l in reversed(range(self.levels)):
                z, log_det = self.injective_block[l](z, log_det, forward=forward)
                z, log_det = self.bijective_block[l](z, log_det, forward=forward)
                z = self.squeeze_block[l](z, forward=forward)
            x = z
            return x


# BIJECTIVE FLOW
class BijectiveFlow(keras.Model):

    def __init__(self, img_shape, num_channels, blocks, levels):
         super(BijectiveFlow, self).__init__()

         self.B, self.H, self.W, self.C = img_shape
         self.input_dimension = self.H * self.W * self.C
         self.num_channels = num_channels
         self.blocks = blocks
         self.levels = levels
         self.squeeze = Squeeze()
         self.bijective_block = []
         self.split = []

         for l in range(levels):
             ops = []
             for o in range(0, blocks):
                 ops.append(RevnetStep(num_channels, type='bijective'))
             self.bijective_block.append(ops)
             if l != levels - 1:
                 self.split.append(Split())
             else:
                 self.split.append(Gaussianize(2 * self.C))

    def latent(self, dim, latent_shape):
        self.input_dimension = dim
        self.B, self.H, self.W, self.C = latent_shape

    def sample_image(self, temperature):
        self.latent_distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.0] * self.input_dimension)
        z = self.latent_distribution.sample() * temperature
        return z

    def call(self, inputs, forward=True, reconstruction=False):

        if forward:
            x = inputs
            pz = 0.0
            log_det = 0.0
            zs_list = []

            for l in range(self.levels - 1):
                #print("----------BIJECTIVE LEVEL----------:", l)
                #print("x", x.shape)
                x = self.squeeze(x, forward=forward)
                #print("Squeeze:", x.shape)
                for op in range(self.blocks):
                    x, log_det = self.bijective_block[l][op](x, log_det, forward=forward)
                #print("Glow:", x.shape)
                x, z_i, log_pz = self.split[l](x, forward=forward)
                #print("Split x:", x.shape)
                #print("Split z_i:", z_i.shape)

                pz += log_pz
                latent_dim = np.prod(z_i.shape[1:])
                zs_list.append(tf.reshape(z_i, [-1, latent_dim]))

            #print("----------LAST BIJECTIVE LEVEL----------|")
            #print("x", x.shape)
            x = self.squeeze(x, forward=forward)
            #print("Squeeze:", x.shape)
            for op in range(self.blocks):
                x, log_det = self.bijective_block[-1][op](x, log_det, forward=forward)
            #print("Glow:", x.shape)
            z_i, parameters = self.split[-1]([tf.zeros_like(x), x], forward=forward)
            #print("Split:", z_i.shape)
            shape = z_i.shape

            mu, sigma = parameters
            log_pz = log_probability(z_i, mu, sigma)
            log_pz = tf.math.reduce_mean((tf.math.reduce_sum(log_pz, axis=[1, 2, 3])), axis=0)

            pz += log_pz
            latent_dim = np.prod(z_i.shape[1:])
            zs_list.append(tf.reshape(z_i, [-1, latent_dim]))
            zs_list = tf.concat(zs_list, axis=1)

            # Loss function
            const = -self.input_dimension * np.log(1 / 256)
            neg_log_likelihood = (-pz - log_det + const) / (np.log(2) * self.input_dimension)
            return zs_list, shape, neg_log_likelihood, pz, log_det

        else:
            log_det = 0.0
            z = inputs
            if reconstruction:
                z = z[0]

            zs_list = []
            start = stop = 0
            for l in range(self.levels - 1):
                stop += self.input_dimension // (2 ** (l + 1))
                zs_list.append(z[start:stop])
                start = stop

            zs_list.append(z[start:])
            z_i = zs_list[-1]
            z_i = tf.reshape(z_i, shape=(1, self.H, self.W, self.C))

            x = self.split[-1]([tf.zeros_like(z_i), z_i], forward=forward)
            for op in reversed(range(self.blocks)):
                x, log_det = self.bijective_block[-1][op](x, log_det, forward=forward)
            x = self.squeeze(x, forward=forward)

            for l in reversed(range(self.levels - 1)):
                x1 = x
                z_i = zs_list[l]
                z_i = tf.reshape(z_i, shape=x1.shape)
                x = self.split[l]([x1, z_i], forward=forward)
                for op in reversed(range(self.blocks)):
                    x, log_det = self.bijective_block[l][op](x, log_det, forward=forward)
                x = self.squeeze(x, forward=forward)

            return x