import numpy as np
import tensorflow as tf

def OOD_detection(nlls_test, nll_trained_model, test_labels, threshold):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0

    FP_indexes = []
    FN_indexes = []
    P = tf.reduce_sum(tf.cast(tf.equal(test_labels, 1), tf.int32)).numpy()
    N = tf.reduce_sum(tf.cast(tf.equal(test_labels, 0), tf.int32)).numpy()

    for i in range(len(nlls_test)):
        c = test_labels[i]
        if np.abs(nlls_test[i] - nll_trained_model) > threshold:
            if c == 1:
                TP += 1
            elif c == 0:
                FP += 1
                FP_indexes.append(i)


        elif np.abs(nlls_test[i] - nll_trained_model) <= threshold:
            if c == 1:
                FN += 1
                FN_indexes.append(i)
            elif c == 0:
                TN += 1

    TPR = TP/P  # sensitivity/recall
    FPR = FP/N  # fall-out
    TNR = TN/N  # specificity
    FNR = FN/P  # miss rate

    gmean = np.sqrt(TPR * (1 - FPR))
    return TPR, FPR, TNR, FNR, gmean, FP_indexes, FN_indexes