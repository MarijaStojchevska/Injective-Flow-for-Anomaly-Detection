import numpy as np
from utils import *
from matplotlib import pyplot as plt

FLAGS, unparsed = flags()
dataset = FLAGS.dataset
categoryName = FLAGS.categoryName

# MNIST testset plotting
def testsetPlotMNIST(test_dataset, test_labels):
    rows = cols = 6
    i = 0
    fig = plt.figure(figsize=(8, 8))
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, r * cols + c + 1)
            plt.imshow(test_dataset[i, :, :, 0], cmap='gray')
            y = test_labels[i]
            i += 1
            plt.title('%i' % y)
            plt.axis('off')

    plt.suptitle("Testset")
    fig.tight_layout()
    plt.savefig('{0}/output_images/testset.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# MVTec dataset plotting
def setPlotMVTEC(dataset, labels, categoryName, type):
    rows = cols = 6
    i = 0
    fig = plt.figure(figsize=(9, 9))
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, r * cols + c + 1)
            if categoryName == "grid" or categoryName == "screw" or categoryName == "zipper":
                plt.imshow(dataset[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(dataset[i, :, :, :])
            if type == "testset":
                y = labels[i]
                plt.title('{0}'.format(y))
            i += 1
            plt.axis('off')
    if type == "testset":
        plt.suptitle("Testset")
        fig.tight_layout()
        plt.savefig('{0}/output_images/testset.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    else:
        plt.suptitle("Trainset")
        fig.tight_layout()
        plt.savefig('{0}/output_images/trainset.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Training examples plotting
def imagesPlot(data, type):
    rows = cols = 4
    i = 0
    fig = plt.figure(figsize=(8, 8))
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, r * cols + c + 1)
            plt.imshow(data[i, :, :, 0], cmap='gray')
            i += 1
            plt.axis('off')

    if type == 'z':
        plt.suptitle("Z")
        plt.savefig('{0}/output_images/latent.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    elif type == 'x':
        plt.suptitle("Reconstructions")
        plt.savefig('{0}/output_images/recon.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Latent samples + reconstructions plotting
def latentPlot(Z):
    fig = plt.figure()
    for z in Z:
        plt.plot(z.numpy()[0], z.numpy()[1], '.')
        plt.title("Z")
    plt.savefig('{0}/output_images/Z.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def reconPlot(data, type):
    rows = 6
    cols = 6
    fig = plt.figure(figsize=(8, 8))
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, r * cols + c + 1)
            plt.imshow(data[r * cols + c][0, :, :, 0], cmap="gray")
            plt.axis("off")

    if type == 'Z_reversed':
        plt.suptitle("Z_reversed")
        plt.savefig('{0}/output_images/Z_reversed.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    elif type == 'X':
        plt.suptitle("X")
        plt.savefig('{0}/output_images/X.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Loss plotting
def lossPlot(train_mse_results, test_mse_results, train_ml_results):
    fig = plt.figure()
    fig, axes = plt.subplots(2, sharex=False, figsize=(12, 8))
    fig.suptitle('Train/Test Loss')

    axes[0].set_ylabel("MSE (Injective Network)")
    axes[0].set_xlabel("Epoch")
    line1, = axes[0].plot(train_mse_results, label="Train Loss")
    line2, = axes[0].plot(test_mse_results, label="Test Loss")
    axes[0].legend(handles=[line1, line2])

    axes[1].set_ylabel("ML (Bijective Network)")
    axes[1].set_xlabel("Epoch")
    axes[1].plot(train_ml_results)

    plt.savefig('{0}/output_images/Loss.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Interpolations plotting
def interpolationsPlot(num_pairs, num_inter, decodings):
    fig = plt.figure()
    for row in range(num_pairs):
        for col in range(num_inter):
            fig.add_subplot(num_pairs, num_inter, row * num_inter + col + 1)
            plt.imshow(decodings[row][col][0, :, :, 0], cmap="gray")
            plt.axis('off')
    plt.suptitle("Interpolation between latent samples")
    plt.savefig('{0}/output_images/Interpolations.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# OOD reconstructions plotting
def reconstructionsPlot(input_data, data, type):
    rows = cols = int(np.sqrt(input_data.shape[0]))
    i = 0
    fig = plt.figure(figsize=(8, 8))
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, r * cols + c + 1)
            if type == 'reconstructions':
                plt.imshow(data[i][0, :, :, 0], cmap='gray')
            elif type == 'original':
                plt.imshow(data[i, :, :, 0], cmap='gray')
            i += 1
            plt.axis('off')

    if type == 'reconstructions':
        plt.suptitle("OOD reconstructions")
        plt.savefig('{0}/output_images/ood_reconstructions.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    elif type == 'original':
        plt.suptitle("Input data")
        plt.savefig('{0}/output_images/ood.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Progress plotting
def progressPlot(progress, epoch_num):
    partial_progress = []
    partial_epoch_num = []
    for i in range(len(progress)):
        if i < 6 or i > len(progress) - 6:
            partial_progress.append(progress[i])
            partial_epoch_num.append(epoch_num[i])

    cols = len(partial_progress)
    fig = plt.figure(figsize=(20, 4))
    for c in range(cols):
        fig.add_subplot(1, cols, c + 1)
        plt.imshow(partial_progress[c][0, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.title('epoch: {}'.format(partial_epoch_num[c]))
    plt.suptitle("Reconstruction progress per epoch")
    plt.savefig('{0}/output_images/Progress.png'.format(FLAGS.path), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


 # AUC-ROC plotting
def rocPlot(tpr1_list, fpr1_list, tpr2_list, fpr2_list, ix1, ix2):
    auc1 = -1 * np.trapz(tpr1_list, fpr1_list)
    auc2 = -1 * np.trapz(tpr2_list, fpr2_list)
    fig = plt.figure()
    plt.plot(fpr1_list, tpr1_list, linestyle=':', color='darkviolet', lw=2, label='Injective Flow (AUC = {0:0.3f})'.format(auc1), clip_on=False)
    plt.plot(fpr2_list, tpr2_list, color='darkorange', lw=2, label='Bijective Flow (AUC = {0:0.3f})'.format(auc2), clip_on=False)
    plt.scatter(fpr1_list[ix1], tpr1_list[ix1], marker='o', color='black', label='Best')
    plt.scatter(fpr2_list[ix2], tpr2_list[ix2], marker='o', color='black')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.savefig('{0}/output_images/ROC-AUC.png'.format(FLAGS.path))
    plt.show()
    plt.close(fig)


# FP plotting
def fpPlot(FP_indexes, test_data, test_labels, original_test, type):
    for i in range(len(FP_indexes)):
        fig = plt.figure(figsize=(8, 8))
        inx = FP_indexes[i]
        fig.add_subplot(2, 1, 1)
        plt.imshow(test_data[inx, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.title('%i' % test_labels[inx])
        fig.add_subplot(2, 1, 2)
        if categoryName == "grid" or categoryName == "screw" or categoryName == "zipper" or dataset == "MNIST":
            plt.imshow(original_test[inx, :, :, 0], cmap='gray')
        else:
            plt.imshow(original_test[inx, :, :, :])
        plt.axis('off')
        fig.tight_layout()
        if type == 0:
            plt.suptitle("False Positives (IF) and their actual class")
            plt.savefig('{0}/FP_IF/{1}.png'.format(FLAGS.path, i), dpi=300, bbox_inches='tight')
        else:
            plt.suptitle("False Positives (BF) and their actual class")
            plt.savefig('{0}/FP_BF/{1}.png'.format(FLAGS.path, i), dpi=300, bbox_inches='tight')
        plt.close(fig)


# FN plotting
def fnPlot(FN_indexes, test_data, test_labels, original_test, type):
    for i in range(len(FN_indexes)):
        fig = plt.figure(figsize=(8, 8))
        inx = FN_indexes[i]
        fig.add_subplot(2, 1, 1)
        plt.imshow(test_data[inx, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.title('%i' % test_labels[inx])
        fig.add_subplot(2, 1, 2)
        if categoryName == "grid" or categoryName == "screw" or categoryName == "zipper" or dataset == "MNIST":
            plt.imshow(original_test[inx, :, :, 0], cmap='gray')
        else:
            plt.imshow(original_test[inx, :, :, :])
        plt.axis('off')
        fig.tight_layout()
        if type == 0:
            plt.suptitle("False Negatives (IF) and their actual class")
            plt.savefig('{0}/FN_IF/{1}.png'.format(FLAGS.path, i), dpi=300, bbox_inches='tight')
        else:
            plt.suptitle("False Negatives (BF) and their actual class")
            plt.savefig('{0}/FN_BF/{1}.png'.format(FLAGS.path, i), dpi=300, bbox_inches='tight')
        plt.close(fig)