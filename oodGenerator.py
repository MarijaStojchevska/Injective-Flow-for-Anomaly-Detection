import numpy as np
import tensorflow as tf


def ood_generator(X_test_mnist, type):
    outliers = X_test_mnist
    oitliers_size = outliers.shape[0]
    output = []

    for i in range(oitliers_size):
        sample = outliers[i, :, :, 0]

        if type == "diagonal":
            sample = sample.numpy()
            np.fill_diagonal(sample, 0.5)
        elif type == "off-diagonal":
            sample = sample.numpy()
            np.fill_diagonal(np.fliplr(sample), 0.5)
        elif type == "cross":
            sample = sample.numpy()
            np.fill_diagonal(sample, 0.5)
            np.fill_diagonal(np.fliplr(sample), 0.5)
        elif type == "spots":
            side = np.random.randint(1, 5)
            if side == 1:
                x = np.random.randint(-0.1, 31)
                y = np.random.randint(28.6, 31)
            elif side == 2:
                x = np.random.randint(28.6, 31)
                y = np.random.randint(-0.1, 31)
            elif side == 3:
                x = np.random.randint(-0.1, 31)
                y = np.random.randint(-0.1, 2.5)
            elif side == 4:
                x = np.random.randint(-0.1, 2.5)
                y = np.random.randint(-0.1, 31)
            img = tf.Variable(sample)
            img[x, y].assign(0.5)
            sample = tf.convert_to_tensor(img)
        else:
            mix_type = np.random.randint(1, 5)
            if mix_type == 1:
                sample = sample.numpy()
                np.fill_diagonal(sample, 0.5)
            elif mix_type == 2:
                sample = sample.numpy()
                np.fill_diagonal(np.fliplr(sample), 0.5)
            elif mix_type == 3:
                sample = sample.numpy()
                np.fill_diagonal(sample, 0.5)
                np.fill_diagonal(np.fliplr(sample), 0.5)
            elif mix_type == 4:
                side = np.random.randint(1, 5)
                if side == 1:
                    x = np.random.randint(-0.1, 31)
                    y = np.random.randint(28.6, 31)
                elif side == 2:
                    x = np.random.randint(28.6, 31)
                    y = np.random.randint(-0.1, 31)
                elif side == 3:
                    x = np.random.randint(-0.1, 31)
                    y = np.random.randint(-0.1, 2.5)
                elif side == 4:
                    x = np.random.randint(-0.1, 2.5)
                    y = np.random.randint(-0.1, 31)
                img = tf.Variable(sample)
                img[x, y].assign(0.5)
                sample = tf.convert_to_tensor(img)


        sample = tf.convert_to_tensor(sample, tf.float32)
        output.append(sample)
    output = tf.reshape(output, (len(output), 32, 32, 1))
    return output