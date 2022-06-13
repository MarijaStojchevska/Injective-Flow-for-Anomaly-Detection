import argparse

def flags():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path',
        type=str,
        default='/scicore/home/rothvo/stojch0001/InjectiveFlow',
        help='path to the main directory')

    parser.add_argument(
        '--dataset',
        type=str,
        default='MVTEC',
        help='which dataset to work with [MNIST, MVTEC]')

    parser.add_argument(
        '--testset_type',
        type=str,
        default='inverted',
        help='which MNIST testset to work with [diagonal, off-diagonal, spots, cross, mixed, inverted, fashion]')

    parser.add_argument(
        '--categoryName',
        type=str,
        default='carpet',
        help='which MVTEC category to work with [bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper]')

    parser.add_argument(
        '--featureExtractor',
        type=str,
        default='densenet',
        help='which feature extractor to work with [densenet, vgg16]')

    parser.add_argument(
        '--runTrain',
        type=bool,
        default=True,
        help='whether to run the training procedure')

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=600,
        help='number of training epochs')

    parser.add_argument(
        '--threshold',
        type=int,
        default=150,
        help='when should ml training begin')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='size of the training batch')

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate')

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed
