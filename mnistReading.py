from plotter import *
from utils import *
from oodGenerator import *
from mlxtend.data import loadlocal_mnist

FLAGS, unparsed = flags()

def mnistReading():
    # Initializing the type of outliers
    type = FLAGS.testset_type

    # Data accessing
    X_train_mnist, Y_train_mnist = loadlocal_mnist(images_path='{0}/MNIST/train-images-idx3-ubyte'.format(FLAGS.path), labels_path='{0}/MNIST/train-labels-idx1-ubyte'.format(FLAGS.path))
    X_test_mnist, Y_test_mnist = loadlocal_mnist(images_path='{0}/MNIST/t10k-images-idx3-ubyte'.format(FLAGS.path), labels_path='{0}/MNIST/t10k-labels-idx1-ubyte'.format(FLAGS.path))
    X_train_fashion, Y_train_fashion = loadlocal_mnist(images_path='{0}/FASHION/train-images-idx3-ubyte'.format(FLAGS.path), labels_path='{0}/FASHION/train-labels-idx1-ubyte'.format(FLAGS.path))
    X_test_fashion, Y_test_fashion = loadlocal_mnist(images_path='{0}/FASHION/t10k-images-idx3-ubyte'.format(FLAGS.path), labels_path='{0}/FASHION/t10k-labels-idx1-ubyte'.format(FLAGS.path))

    # Data preprocessing
    def preprocess(x):
        x = x / 255.0 - 0.5
        x = x + tf.random.uniform(shape=tf.shape(x), minval=0, maxval=1 / 256.0, dtype=x.dtype)
        return x

    # Data reading
    def reading(X_train, X_test):
        X_train = X_train.reshape(60000, 28, 28)
        X_test = X_test.reshape(10000, 28, 28)
        X_train = np.expand_dims(X_train, axis=3)
        X_test = np.expand_dims(X_test, axis=3)
        X_train = X_train[0:, :, :, :]
        X_test = X_test[0:, :, :, :]
        X_train = np.pad(X_train, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))
        X_test = np.pad(X_test, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        return X_train, X_test

    # Load data sets
    X_train_mnist, X_test_mnist = reading(X_train_mnist, X_test_mnist)
    X_train_fashion, X_test_fashion = reading(X_train_fashion, X_test_fashion)

    # Preprocess data sets
    X_train_mnist = preprocess(X_train_mnist)
    X_test_mnist = preprocess(X_test_mnist)
    X_train_fashion = preprocess(X_train_fashion)
    X_test_fashion = preprocess(X_test_fashion)

    # Select test and validation set
    X_test_1 = X_test_mnist[0:2500]
    X_test_2 = X_test_fashion[2500:5000]
    X_val = X_test_mnist[5000:10000]
    trainset = tf.convert_to_tensor(X_train_mnist, tf.float32)
    fashion = tf.convert_to_tensor(X_train_fashion, tf.float32)
    valset = tf.convert_to_tensor(X_val, tf.float32)

    # Choose testset type
    if type != "fashion":
        inliers = X_test_mnist[0:2500]
        if type == 'inverted':
            outliers = np.abs(X_test_mnist[2500:5000] - 255)  # color inversion
        else:
            outliers = ood_generator(X_test_mnist[2500:5000], type=type)
        X_test_1 = inliers
        X_test_2 = outliers

    X_test_1 = tf.convert_to_tensor(X_test_1, tf.float32)
    X_test_2 = tf.convert_to_tensor(X_test_2, tf.float32)

    # Test data and labels generation
    test_dataset = tf.concat([X_test_1, X_test_2], 0)
    n = len(X_test_1)+len(X_test_2)
    test_labels = []
    for i in range(0, n):
        if (i >= n/2):
            test_labels.append(1)  # outlier
        else:
            test_labels.append(0)  # inlier

    indices = tf.range(start=0, limit=n, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    test_dataset = tf.gather(test_dataset, shuffled_indices)
    test_labels = tf.gather(test_labels, shuffled_indices)
    testsetPlotMNIST(test_dataset, test_labels)
    testset = [test_dataset, test_labels]

    t = np.random.randint(0, len(fashion)-36)

    return trainset[0:30000], testset, test_dataset, valset, fashion[t:t+36]
