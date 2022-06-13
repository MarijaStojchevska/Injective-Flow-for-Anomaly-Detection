import cv2
import keras
import pathlib
from plotter import *
import tensorflow as tf
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage.exposure import histogram
from keras.preprocessing.image import ImageDataGenerator


FLAGS, unparsed = flags()
bottle = {"t1": 0.99, "t2": 13, "sign": '<'}
capsule = {"t1": 0.47, "t2": 11.9, "sign": '<'}
pill = {"t1": 0.12, "t2": 12.2, "sign": '>'}
metal_nut = {"t1": 0.11, "t2": 12, "sign": '>'}
toothbrush = {"t1": 0.09, "t2": 12.2, "sign": '>'}

def mvtecReading():
    categoryName = FLAGS.categoryName
    pathTrain = "{0}/MVTEC/{1}/train/good/".format(FLAGS.path, categoryName)
    pathTest = "{0}/MVTEC/{1}/test/".format(FLAGS.path, categoryName)

    bad_background = ['bottle', 'capsule', 'pill', 'metal_nut', 'toothbrush']
    no_channels = ['grid', 'screw', 'zipper']

    if categoryName == "bottle":
        defects = ["broken_large", "broken_small", "contamination", "good"]
    elif categoryName == "cable":
        defects = ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation", "good", "missing_cable", "missing_wire", "poke_insulation"]
    elif categoryName == "capsule":
        defects = ["crack", "faulty_imprint", "good", "poke", "scratch", "squeeze"]
    elif categoryName == "carpet":
        defects = ["color", "cut", "good", "hole", "metal_contamination", "thread"]
    elif categoryName == "grid":
        defects = ["bent", "broken", "glue", "good", "metal_contamination", "thread"]
    elif categoryName == "hazelnut":
        defects = ["crack", "cut", "good", "hole", "print"]
    elif categoryName == "leather":
        defects = ["color", "cut", "fold", "glue", "good", "poke"]
    elif categoryName == "metal_nut":
        defects = ["bent", "color", "flip", "good", "scratch"]
    elif categoryName == "pill":
        defects = ["color", "combined", "contamination", "crack", "faulty_imprint", "good", "pill_type", "scratch"]
    elif categoryName == "screw":
        defects = ["good", "manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]
    elif categoryName == "tile":
        defects = ["crack", "glue_strip", "good", "gray_stroke", "oil", "rough"]
    elif categoryName == "toothbrush":
        defects = ["defective", "good"]
    elif categoryName == "transistor":
        defects = ["bent_lead", "cut_lead", "damaged_case", "good", "misplaced"]
    elif categoryName == "wood":
        defects = ["color", "combined", "good", "hole", "liquid", "scratch"]
    elif categoryName == "zipper":
        defects = ["broken_teeth", "combined", "fabric_border", "fabric_interior", "good", "rough", "split_teeth", "squeezed_teeth"]

    # reading images that do not require background preprocessing
    def dataReading(path):
        originalImages = []
        for p in pathlib.Path(path).iterdir():
            if categoryName not in no_channels:
                img = plt.imread(p)
            elif categoryName in no_channels:
                img = cv2.imread(str(p))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.zeros_like(img)
                img[:, :, 0] = gray
                img[:, :, 1] = gray
                img[:, :, 2] = gray
            shape = img.shape
            originalImages.append(img)
        originalImages = np.array(originalImages)
        originalImages = originalImages.reshape(len(originalImages), shape[0], shape[1], shape[2])
        return originalImages, shape

    # data standardization
    def dataStandardization(dataset):
        dataset = dataset / 255.
        mean = dataset.mean(axis=(0, 1, 2))
        std = dataset.std(axis=(0, 1, 2))

        dataset[..., 0] -= mean[0]
        dataset[..., 1] -= mean[1]
        dataset[..., 2] -= mean[2]

        dataset[..., 0] /= std[0]
        dataset[..., 1] /= std[1]
        dataset[..., 2] /= std[2]
        return dataset

    # rgb to gray
    def image2gray(path):
        images = []
        images_grayscale = []
        for p in pathlib.Path(path).iterdir():
            img = plt.imread(p)
            gray = rgb2gray(img)
            images.append(img)
            images_grayscale.append(gray)
        return images, images_grayscale

    # mask application
    def apply_mask(img, mask):
        masked_img = img.copy()
        masked_img[~mask] = -0.5
        return masked_img

    # image masking
    def maskImages(images, images_grayscale, t1, t2, sign):
        originalImages = []
        for i in range(len(images_grayscale)):
            if sign == '<':
                mask = images_grayscale[i] < t1
            else:
                mask = images_grayscale[i] > t1

            apply_mask(images[i], mask)
            mask_patched = ndi.binary_fill_holes(mask)
            apply_mask(images[i], mask_patched)

            label_objects, nb_labels = ndi.label(mask_patched)
            sizes = np.bincount(label_objects.ravel())
            num_objects_limit = np.exp(t2)
            '''
            plt.hist(np.log1p(sizes), bins=100)
            plt.yscale('log')
            plt.axvline(np.log1p(num_objects_limit), color='red')
            plt.show()
            '''
            mask_sizes = sizes > num_objects_limit
            mask_sizes[0] = 0
            mask_patched = mask_sizes[label_objects]
            masked_img_no_blips = apply_mask(images[i], mask_patched)
            shape = masked_img_no_blips.shape
            originalImages.append(masked_img_no_blips)
        originalImages = np.array(originalImages)
        originalImages = originalImages.reshape(len(originalImages), shape[0], shape[1], shape[2])
        return originalImages, shape

    # select threshold from the histogram of all grayscale images and use it to mask an image
    def preprocessing(path, t1, t2, sign):
        images, images_grayscale = image2gray(path)
        '''
        plt.figure(figsize=(10, 5))
        fig, axes = plt.subplots(1, figsize=(12, 8))
        plt.title('Histogram for all training gray scaled images')
        axes.set_ylabel("Frequency")
        axes.set_xlabel("Intensity (gray value)")
        for i in range(len(images_grayscale)):
            hist, hist_centers = histogram(images_grayscale[i])
            axes.plot(hist_centers, hist, marker='+', linestyle='-', alpha=0.5)
        threshold = t1
        plt.axvline(threshold, color='red')
        plt.show()
        '''
        originalImages, shape = maskImages(images, images_grayscale, t1, t2, sign)
        return originalImages, shape

    # reading images that do require background preprocessing
    def objectPreprocessing(path):
        originalImages, shape = preprocessing(path=path, t1=eval(categoryName)["t1"], t2=eval(categoryName)["t2"], sign=eval(categoryName)["sign"])
        return originalImages, shape

    # data augmentor
    def dataAugmentation(image):
        shape = image.shape
        img = image.reshape(1, shape[0], shape[1], shape[2])
        img = img.astype('float32')
        a = 2 * np.pi
        count = 20
        theta = a * np.random.uniform(low=0.0, high=1.0, size=count) - np.pi
        for t in theta:
            angle = t * (180 / np.pi)
            rotation = ImageDataGenerator(rotation_range=angle)
            iterator = rotation.flow(img, batch_size=1)
            batch = iterator.next()
            batch = batch.reshape(shape[0], shape[1], shape[2])
            augmented_set.append(batch)
        return augmented_set, count

    # data shuffler
    def shuffle(inputSet, inputLabels, classificationLabels):
        indices = tf.range(start=0, limit=len(inputSet), dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        outputSet = tf.gather(inputSet, shuffled_indices)
        outputLables = tf.gather(inputLabels, shuffled_indices)
        classLabels = tf.gather(classificationLabels, shuffled_indices)

        outputSet = np.array(outputSet)
        outputSet = outputSet.reshape(len(outputSet), outputSet.shape[1], outputSet.shape[2], outputSet.shape[3])
        return outputSet, outputLables, classLabels

    # image scaling
    def scaling(factor, input):
        input = np.array(input)
        width = height = int(factor)
        resized = []
        for i in range(len(input)):
            resized.append(cv2.resize(input[i], (width, height), interpolation=cv2.INTER_AREA))
        resized = np.array(resized)
        resized = resized.reshape(len(resized), factor, factor, input.shape[3])
        return resized

    # feature extractor
    def featureExtractor(input, input_shape):
        m1 = keras.models.load_model("{0}/{1}/model_{2}.h5".format(FLAGS.path, FLAGS.featureExtractor, input_shape[1]))
        model = keras.Model(inputs=m1.inputs, outputs=m1.layers[-2].output)
        extracted_features = model.predict(input)
        extracted_features = tf.convert_to_tensor(extracted_features, tf.float32)
        return extracted_features

    # feature extraction from images
    def rescalingFeatures(scale, factor, train_images, test_images, val_images):
        if scale == True:
            train_images = scaling(factor, train_images)
            test_images = scaling(factor, test_images)
            val_images = scaling(factor, val_images)

        train_features = featureExtractor(input=train_images, input_shape=train_images.shape)
        test_features = featureExtractor(input=test_images, input_shape=test_images.shape)
        val_features = featureExtractor(input=val_images, input_shape=val_images.shape)

        train_features = tf.math.reduce_mean(train_features, axis=[3], keepdims=True)
        test_features = tf.math.reduce_mean(test_features, axis=[3], keepdims=True)
        val_features = tf.math.reduce_mean(val_features, axis=[3], keepdims=True)

        if FLAGS.featureExtractor == 'vgg16':
            scaleFactor = 64
        else:
            scaleFactor = 32

        if train_features.shape[1] != scaleFactor:
            train_features = scaling(scaleFactor, train_features)
            test_features = scaling(scaleFactor, test_features)
            val_features = scaling(scaleFactor, val_features)

        train_features = tf.convert_to_tensor(train_features[:, :, :, :], tf.float32)
        test_features = tf.convert_to_tensor(test_features[:, :, :, :], tf.float32)
        val_features = tf.convert_to_tensor(val_features[:, :, :, :], tf.float32)
        return train_features, test_features, val_features




    # reading training images
    if categoryName not in bad_background:
        train_images, train_shape = dataReading(path=pathTrain)
    else:
        train_images, train_shape = objectPreprocessing(path=pathTrain)
    print("Input size of training set: ", len(train_images))

    # data augmentation for training examples
    augmented_set = []
    for i in range(len(train_images)):
        augmented_set.append(train_images[i])
        augmented_set, _ = dataAugmentation(train_images[i])
    train_images, _, _ = shuffle(augmented_set, augmented_set, augmented_set)
    print("Size of augmented training set: ", len(train_images))
    setPlotMVTEC(train_images, [], categoryName=categoryName, type="trainset")

    # data standardization for training examples
    train_images = dataStandardization(train_images)

    val_images = train_images[0:30]
    train_images = train_images[30:]
    test_labels = []
    classification_labels = []

    # testset reading
    for d in defects:
        count = 0
        dir = pathTest+"{0}".format(d)
        if categoryName not in bad_background:
            images, shape = dataReading(path=dir)
        else:
            images, shape = objectPreprocessing(path=dir)

        for path in pathlib.Path(dir).iterdir():
            if path.is_file():
                count += 1
        if d == "good":
            type = str(d + ' - 0')
            test_labels.extend([type]*count)
            classification_labels.extend([0]*count)
        else:
            type = str(d + ' - 1')
            test_labels.extend([type]*count)
            classification_labels.extend([1]*count)

        if d == defects[0]:
                test_images = tf.convert_to_tensor(images[:, :, :, :], tf.float32)
        else:
                test_images = np.concatenate((test_images, images[:, :, :, :]), axis=0)
    print("Input size of test set: ", len(test_images))
    setPlotMVTEC(test_images, test_labels, categoryName=categoryName, type="testset")

    # data augmentation for test examples
    augmented_set = []
    for i in range(len(test_images)):
        augmented_set.append(test_images[i])
        augmented_set, test_aug_factor = dataAugmentation(test_images[i])
    augmented_set = np.array(augmented_set)
    augmented_set = augmented_set.reshape(len(augmented_set), test_images.shape[1], test_images.shape[2], test_images.shape[3])
    print("Size of augmented test set: ", len(augmented_set))

    original_test_images = test_images
    test_images = augmented_set

    # data standardization for test examples
    test_images = dataStandardization(test_images)

    print("Train data: ", len(train_images))
    print("Test data: ", len(test_images))
    print("Validation data: ", len(val_images))

    # scaling and feature extraction for all datasets
    original_train_features, original_test_features, original_val_features = rescalingFeatures(scale=False, factor=None, train_images=train_images, test_images=test_images, val_images=val_images)
    recondata = original_test_features
    recondata, _, _ = shuffle(recondata, recondata, recondata)
    t = np.random.randint(0, len(recondata) - 36)

    return original_train_features, [original_test_features, classification_labels], original_test_images, original_val_features, recondata[t:t+36], test_aug_factor