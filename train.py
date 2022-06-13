import os
import shutil
from model import *
from time import time
from evaluation import *
from mnistReading import *
from mvtecReading import *

FLAGS, unparsed = flags()
dataset = FLAGS.dataset

if dataset == 'MNIST':
    trainset, testset, original_test, valset, recondata = mnistReading()
elif dataset == "MVTEC":
    trainset, testset, original_test, valset, recondata, test_aug_factor = mvtecReading()
img_shape = (1, trainset.shape[1], trainset.shape[2], trainset.shape[3])


def train(run_train, num_epochs, threshold, batch_size, dataset, lr, exp_path):
    print('Experiment setup:')
    print('---> num_epochs: {}'.format(num_epochs))
    print('---> batch_size: {}'.format(batch_size))
    print('---> dataset: {}'.format(dataset))
    print('---> Learning rate: {}'.format(lr))
    print('---> experiment path: {}'.format(exp_path))
    training_images = trainset
    testing_images = testset[0]
    print('Dataset is loaded: training and test dataset shape: {} {}'.format(training_images.shape, testing_images.shape))

    if os.path.exists(os.path.join(exp_path, 'logs')):
        shutil.rmtree(os.path.join(exp_path, 'logs'))

    MSE_train_log_dir = os.path.join(exp_path, 'logs', 'MSE_train')
    MSE_train_summary_writer = tf.summary.create_file_writer(MSE_train_log_dir)
    MSE_train_loss_metric = tf.keras.metrics.Mean('MSE_train_loss', dtype=tf.float32)

    MSE_test_log_dir = os.path.join(exp_path, 'logs', 'MSE_test')
    MSE_test_summary_writer = tf.summary.create_file_writer(MSE_test_log_dir)
    MSE_test_loss_metric = tf.keras.metrics.Mean('MSE_test_loss')

    ML_log_dir = os.path.join(exp_path, 'logs', 'ML')
    ML_summary_writer = tf.summary.create_file_writer(ML_log_dir)
    ML_loss_metric = tf.keras.metrics.Mean('ML_loss', dtype=tf.float32)

    pz_log_dir = os.path.join(exp_path, 'logs', 'pz')
    pz_summary_writer = tf.summary.create_file_writer(pz_log_dir)
    pz_metric = tf.keras.metrics.Mean('pz', dtype=tf.float32)

    jacobian_log_dir = os.path.join(exp_path, 'logs', 'jacobian')
    jacobian_summary_writer = tf.summary.create_file_writer(jacobian_log_dir)
    jacobian_metric = tf.keras.metrics.Mean('jacobian', dtype=tf.float32)

    optimizerIF = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1)
    optimizerBF = tf.keras.optimizers.Adam(learning_rate=lr)

    time_vector = np.zeros([num_epochs, 1])
    IF = InjectiveFlow(img_shape=img_shape, num_channels=500, levels=3)  # injective flow
    BF = BijectiveFlow(img_shape=img_shape, num_channels=2048, blocks=8, levels=1)  # bijective flow

    train_mse_results = []
    test_mse_results = []
    train_ml_results = []
    progress = []
    epoch_num = []

    ckpt = tf.train.Checkpoint(IF=IF, optimizerIF=optimizerIF, BF=BF, optimizerBF=optimizerBF)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint).expect_partial()

    def train_step_mse(sample, epoch):
        with tf.GradientTape() as tape:
            MSE = tf.keras.losses.MeanSquaredError()
            MSE_z = tf.keras.losses.MeanSquaredError()

            z = IF(sample, forward=True)
            IF.latent(z.shape)
            recon = IF(z, forward=False)
            recon_z = IF(recon, forward=True)

            if epoch == threshold - 1:
                imagesPlot(z, type='z')
                imagesPlot(recon, type='x')

            mse_loss = MSE(sample, recon)
            mse_z = MSE_z(z, recon_z)  # Added for stability
            loss = mse_loss + mse_z

            variables = tape.watched_variables()
            grads = tape.gradient(loss, variables)
            optimizerIF.apply_gradients(zip(grads, variables))
        return loss

    @tf.function
    def train_step_ml(x):
        with tf.GradientTape() as tape:
            z, shape, loss, pz, logdet = BF(x, forward=True)
            BF.latent(np.prod(z.shape[1:]), shape)
            variables = tape.watched_variables()
            grads = tape.gradient(loss, variables)
            optimizerBF.apply_gradients(zip(grads, variables))
        return loss, pz, logdet


    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    Ntrain = len(training_images)
    Ntest = len(testing_images)

    if run_train:
        for epoch in range(num_epochs):
            epoch_start = time()

            training_images = tf.random.shuffle(training_images)
            num_iters = Ntrain // batch_size
            for i in range(num_iters):
                if epoch < threshold:
                    t = np.random.randint(0, num_iters)
                    x = training_images[t * batch_size:(t + 1) * batch_size]
                    mse_loss = train_step_mse(x, epoch)
                    ml_loss = np.Inf
                    logdet = 0.0
                    pz = 0.0
                else:
                    if threshold == 0:
                        mse_loss = 0
                    t = np.random.randint(0, num_iters)
                    x = training_images[t * batch_size:(t + 1) * batch_size] 
                    z_batch = IF(x, forward=True) # IF-BF transition
                    ml_loss, pz, logdet = train_step_ml(z_batch)

            testing_images = tf.random.shuffle(testing_images)
            num_iters = Ntest // batch_size
            for i in range(num_iters):
                t = np.random.randint(0, num_iters)
                input_batch = testing_images[t * batch_size:(t + 1) * batch_size]
                # MSE for test images
                z = IF(input_batch, forward=True)
                x = IF(z, forward=False)
                test_mse = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.square(input_batch - x), axis=[1, 2, 3])) / tf.math.sqrt(tf.reduce_sum(tf.square(input_batch), axis=[1, 2, 3])))

            #Progress tracker
            if epoch == threshold or (epoch > threshold and epoch % 2 == 0):
                epoch_num.append(epoch)
                if epoch == threshold:
                    example_instance = BF.sample_image(temperature=0.7)
                x = BF(example_instance, forward=False)
                recon = IF(x, forward=False, reconstruction=True)
                progress.append(recon)

            MSE_train_loss_metric.update_state(mse_loss)
            MSE_test_loss_metric.update_state(test_mse)
            ML_loss_metric.update_state(ml_loss)
            pz_metric.update_state(pz)
            jacobian_metric.update_state(logdet)

            train_mse_results.append(MSE_train_loss_metric.result().numpy())
            test_mse_results.append(MSE_test_loss_metric.result().numpy())
            train_ml_results.append(ML_loss_metric.result().numpy())

            # Saving logs
            with MSE_train_summary_writer.as_default():
                tf.summary.scalar(
                    'MSE_train', MSE_train_loss_metric.result(), step=epoch)

            with MSE_test_summary_writer.as_default():
                tf.summary.scalar(
                    'MSE_test', MSE_test_loss_metric.result(), step=epoch)

            with ML_summary_writer.as_default():
                tf.summary.scalar(
                    'ML_loss', ML_loss_metric.result(), step=epoch)

            with pz_summary_writer.as_default():
                tf.summary.scalar(
                    'pz', pz_metric.result(), step=epoch)

            with jacobian_summary_writer.as_default():
                tf.summary.scalar(
                    'jacobian', jacobian_metric.result(), step=epoch)

            print("Epoch {:03d}: MSE train: {:.3f} / MSE test: {:.3f} / ML Loss: {:.3f} ".format(epoch, MSE_train_loss_metric.result().numpy(), MSE_test_loss_metric.result().numpy(),  ML_loss_metric.result().numpy()))
            MSE_train_loss_metric.reset_states()
            MSE_test_loss_metric.reset_states()
            ML_loss_metric.reset_states()
            pz_metric.reset_states()
            jacobian_metric.reset_states()

            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
            epoch_end = time()
            time_vector[epoch] = epoch_end - epoch_start
            np.save(os.path.join(exp_path, 'time_vector.npy'), time_vector)
            print('epoch time:{}'.format(time_vector[epoch]))

    else:
        x = IF(training_images[0:1], forward=True)
        IF.latent(x.shape)
        z, shape, _, _, _ = BF(x, forward=True)
        BF.latent(np.prod(z.shape[1:]), shape)


    # Loss plotting
    lossPlot(train_mse_results, test_mse_results, train_ml_results)


    # Sampling from a latent distribution
    Z = []
    Z_reversed = []
    X = []
    rows = 6
    cols = 6
    for r in range(rows):
        for c in range(cols):
            z = BF.sample_image(temperature=0.7)
            z_reversed = BF(z, forward=False)
            x = IF(z_reversed, forward=False, reconstruction=True)
            Z.append(z)
            Z_reversed.append(z_reversed)
            X.append(x)
    latentPlot(Z)
    reconPlot(Z_reversed, type='Z_reversed')
    reconPlot(X, type='X')


    # Interpolation between latent samples
    num_pairs = 8
    pairs = []
    for i in range(num_pairs):
        t = i
        x1 = training_images[t:t + 1]
        x1 = IF(x1, forward=True)
        IF.latent(x1.shape)
        z1, shape, _, _, _ = BF(x1, forward=True)
        shape = tf.concat(shape, axis=1)
        dim = np.prod(z1.shape[1:])
        BF.latent(dim, shape)

        t += 1
        x2 = training_images[t:t + 1]
        x2 = IF(x2, forward=True)
        IF.latent(x2.shape)
        z2, shape, _, _, _ = BF(x2, forward=True)
        shape = tf.concat(shape, axis=1)
        dim = np.prod(z2.shape[1:])
        BF.latent(dim, shape)
        pairs.append([z1, z2])

    num_inter = 8
    interpolations = []
    for pair in pairs:
        z1, z2 = pair[0], pair[1]
        inter_per_pair = []
        alpha = 0
        for i in range(num_inter):
            inter_per_pair.append((1 - alpha) * z1 + alpha * z2)
            alpha += 1 / (num_inter - 1.0)
        interpolations.append(inter_per_pair)

    decodings = []
    for list in interpolations:
        imgs = []
        for i, z in enumerate(list):
            x = BF(z, forward=False, reconstruction=True)
            x = IF(x, forward=False, reconstruction=True)
            imgs.append(x)
        decodings.append(imgs)
    interpolationsPlot(num_pairs, num_inter, decodings)


    # OOD sample reconstruction
    ood_reconstructions = []
    for t in range(recondata.shape[0]):
        x = recondata[t:t + 1]
        # forward step (latent)
        x = IF(x, forward=True)
        IF.latent(x.shape)
        z, shape, log_likelihood, pz, logdet = BF(x, forward=True)
        shape = tf.concat(shape, axis=1)
        dim = np.prod(z.shape[1:])
        BF.latent(dim, shape)
        # backward step (reconstruction)
        z = BF(z, forward=False, reconstruction=True)
        x = IF(z, forward=False, reconstruction=True)
        ood_reconstructions.append(x)
    reconstructionsPlot(recondata, ood_reconstructions, type='reconstructions')
    reconstructionsPlot(recondata, recondata, type='original')
    progressPlot(progress, epoch_num)


    # Calculating the number of trainable parametrs
    input_batch = training_images[0:batch_size]
    with tf.GradientTape() as tape:
        z_batch = IF(input_batch, forward=True)
        IF.latent(z_batch.shape)
        variablesIF = tape.watched_variables()
    with tf.GradientTape() as tape:
        z, shape, log_likelihood, pz, logdet = BF(z_batch, forward=True)
        shape = tf.concat(shape, axis=1)
        dim = np.prod(z.shape[1:])
        BF.latent(dim, shape)
        variablesBF = tape.watched_variables()
    parameters_IF = np.sum([np.prod(v.get_shape().as_list()) for v in variablesIF])
    parameters_BF = np.sum([np.prod(v.get_shape().as_list()) for v in variablesBF])
    print("----------------------------------------------------------------")
    print('Number of trainable_parameters of IF: {}'.format(parameters_IF))
    print('Number of trainable_parameters of BF: {}'.format(parameters_BF))
    print('Total number of trainable_parameters: {}'.format(parameters_IF + parameters_BF))


    # Computing the rectangular Jacobian matrix across the entire network
    def model_batch_jacobian(x):
        with tf.GradientTape() as tapeJ:
            tapeJ.watch(x)
            y = IF(x, forward=True)
            z, shape, nll_bijective, pz, logdet = BF(y, forward=True)
            dim = np.prod(z.shape[1:])
        jacobian = tapeJ.batch_jacobian(z, x)
        return jacobian, dim, pz, nll_bijective


    # Computing the log determinant of a rectangular Jacobian
    def model_logdet(x):
        jacobian, dim, pz, nll_bijective = model_batch_jacobian(x)
        jj = tf.linalg.matmul(jacobian, jacobian, transpose_a=True, transpose_b=False)
        jj_det = tf.math.reduce_mean(tf.linalg.det(jj))
        jj_logdet = 0.5 * tf.math.log(jj_det)
        return jj_logdet, dim, pz, nll_bijective


    # Evaluating Negative Log Likelihood across the entire network
    def likelihoodEvaluation(x):
        jj_logdet, dim, pz, nll_bijective = model_logdet(x)
        const = -dim * np.log(1 / 256)
        likelihood1 = (-pz - jj_logdet + const) / (np.log(2) * dim)
        print("Negative log-likelihood (injective flow): ", likelihood1.numpy())
        likelihood2 = nll_bijective
        print("Negative log-ikelihood (bijective flow): ", likelihood2.numpy())
        return likelihood1.numpy(), likelihood2.numpy()


    # Computing NLL per batch for a given dataset
    def NLLs(dataset, batch_size):
        batches = int(dataset.shape[0] / batch_size)
        nlls_model = []
        nlls_bijective = []
        for b in range(batches):
            nll1, nll2 = likelihoodEvaluation(dataset[b * batch_size:(b + 1) * batch_size])
            nlls_model.append(nll1)
            nlls_bijective.append(nll2)
        return nlls_model, nlls_bijective


    # Out of distribution data detection - OOD
    train_nll_1, train_nll_2 = NLLs(trainset[0:100], batch_size=10)
    train_nll_1 = tf.reduce_mean(train_nll_1)
    train_nll_2 = tf.reduce_mean(train_nll_2)
    print("Average nll_1: ",  train_nll_1)
    print("Average nll_2: ", train_nll_2)


    # Creating data dependent thresholds for comparison
    def get_threshold(valset, train_nll_1, train_nll_2, K, batch_size):
        thresholds_1 = []
        thresholds_2 = []
        for k in range(K):
            sample_indx = np.random.choice(valset.shape[0], size=batch_size, replace=False)
            val_nll_1, val_nll_2 = NLLs(valset[sample_indx[0]:sample_indx[0]+1], batch_size=batch_size)
            thresholds_1.append(np.abs(val_nll_1 - train_nll_1))
            thresholds_2.append(np.abs(val_nll_2 - train_nll_2))
        return thresholds_1, thresholds_2

    if dataset == "MVTEC":
        K = len(valset)
    else:
        K = 50

    thresholds_1, thresholds_2 = get_threshold(valset, train_nll_1, train_nll_2, K=K, batch_size=1)
    t1 = np.quantile(thresholds_1, 0.99)
    t2 = np.quantile(thresholds_2, 0.99)
    
    def roc_thresholds(t):
        thresholds = []
        lower_bound = 0
        upper_bound = t*5
        thresholds.append(lower_bound)

        while lower_bound < t:
            lower_bound += 0.01
            thresholds.append(lower_bound)
        thresholds.append(t)

        while t < upper_bound:
            t += 0.01
            thresholds.append(t)
        thresholds.append(upper_bound)
        return thresholds

    thresholds_1 = roc_thresholds(t1)
    thresholds_2 = roc_thresholds(t2)
    thresholds_1.sort()
    thresholds_2.sort()


    # Computing the evaluation metrices for both models
    test_data = testset[0]
    test_labels = testset[1]

    if dataset == 'MNIST':
        test_nll_1, test_nll_2 = NLLs(test_data, batch_size=1)
    else:
        test_nll_1, test_nll_2 = NLLs(test_data, batch_size=1)

        n = test_aug_factor+1
        step = test_aug_factor+1
        final_list_1 = []
        final_list_2 = []

        for i in range(0, len(test_nll_1), step):
            nll1 = 0
            nll2 = 0
            for j in range(i, n):
                nll1 += test_nll_1[j]
                nll2 += test_nll_2[j]
            final_list_1.append(nll1 / step)
            final_list_2.append(nll2 / step)
            n += step
        test_nll_1 = final_list_1
        test_nll_2 = final_list_2


    tpr1_list = []
    fpr1_list = []
    tnr1_list = []
    fnr1_list = []

    tpr2_list = []
    fpr2_list = []
    tnr2_list = []
    fnr2_list = []

    gmeans_1 = []
    gmeans_2 = []

    for t in thresholds_1:
        TPR_1, FPR_1, TNR_1, FNR_1, gmean_1, _, _ = OOD_detection(test_nll_1, train_nll_1.numpy(), test_labels, t)
        tpr1_list.append(TPR_1)
        fpr1_list.append(FPR_1)
        tnr1_list.append(TNR_1)
        fnr1_list.append(FNR_1)
        gmeans_1.append(gmean_1)

    for t in thresholds_2:
        TPR_2, FPR_2, TNR_2, FNR_2, gmean_2, _, _ = OOD_detection(test_nll_2, train_nll_2.numpy(), test_labels, t)
        tpr2_list.append(TPR_2)
        fpr2_list.append(FPR_2)
        tnr2_list.append(TNR_2)
        fnr2_list.append(FNR_2)
        gmeans_2.append(gmean_2)


    # Locating the index of the largest g-mean
    ix1 = np.argmax(gmeans_1)
    ix2 = np.argmax(gmeans_2)
    print('Best IF Threshold=%f, G-Mean=%.3f --> [TPR:%f, FPR:%f, TNR:%f, FNR:%f]' % (thresholds_1[ix1], gmeans_1[ix1], tpr1_list[ix1], fpr1_list[ix1], tnr1_list[ix1], fnr1_list[ix1]))
    print('Best BF Threshold=%f, G-Mean=%.3f --> [TPR:%f, FPR:%f, TNR:%f, FNR:%f]' % (thresholds_2[ix2], gmeans_2[ix2], tpr2_list[ix2], fpr2_list[ix2], tnr2_list[ix2], fnr2_list[ix2]))

    _, _, _, _, _, FP_indexes_1, FN_indexes_1 = OOD_detection(test_nll_1, train_nll_1.numpy(), test_labels, thresholds_1[ix1])
    _, _, _, _, _, FP_indexes_2, FN_indexes_2 = OOD_detection(test_nll_2, train_nll_2.numpy(), test_labels, thresholds_2[ix2])

    print(FP_indexes_1)
    print(FN_indexes_1)
    print(FP_indexes_2)
    print(FN_indexes_2)

    # Plotting AUC-ROC
    rocPlot(tpr1_list, fpr1_list, tpr2_list, fpr2_list, ix1, ix2)

    if dataset == 'MVTEC':
        plotData = []
        for i in range(0, len(test_data), test_aug_factor + 1):
            plotData.append(test_data[i])
        plotData = np.array(plotData)
        plotData = plotData.reshape(len(plotData), plotData.shape[1], plotData.shape[2], plotData.shape[3])
        test_data = plotData

    # Plotting images classified as FP
    if len(FP_indexes_1) > 0:
        fpPlot(FP_indexes_1, test_data, test_labels, original_test, type=0)
    if len(FP_indexes_2) > 0:
        fpPlot(FP_indexes_2, test_data, test_labels, original_test, type=1)

    # Plotting images classified as FN
    if len(FN_indexes_1) > 0:
        fnPlot(FN_indexes_1, test_data, test_labels, original_test, type=0)
    if len(FN_indexes_2) > 0:
        fnPlot(FN_indexes_2, test_data, test_labels, original_test, type=1)



if __name__ == '__main__':
    train(run_train=True,
          num_epochs=FLAGS.num_epochs,
          threshold=FLAGS.threshold,
          batch_size=FLAGS.batch_size,
          dataset=dataset,
          lr=FLAGS.lr,
          exp_path="{0}/saved_models/".format(FLAGS.path))
