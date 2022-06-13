Usage: train.py [--path PATH] [--dataset DATASET]
                [--testset_type TESTSET_TYPE] [--categoryName CLASSNAME]
                [--featureExtractor FEATUREEXTRACTOR] [--runTrain RUNTRAIN]
                [--num_epochs NUM_EPOCHS] [--threshold THRESHOLD]
                [--batch_size BATCH_SIZE] [--lr LR]

Arguments:
--path           path to the main directory
--dataset        which dataset to work with [MNIST, MVTEC]
--testset_type   which MNIST testset to work with [diagonal, off-diagonal, spots, cross, mixed, inverted, fashion]
--categoryName   which MVTEC category to work with [bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper]
--featureExtractor which feature extractor to use [densenet, vgg16]
--runTrain       whether to run the training procedure
--num_epochs     number of training epochs
--threshold      when should the ml training begin
--batch_size     size of the training batch
--lr             learning rate



Instructions for accessing the datasets:
The whole MVTec AD dataset can be downloaded from the official MVTec website: https://www.mvtec.com/company/research/datasets/mvtec-ad
The official link for downloading the MNIST dataset: http://yann.lecun.com/exdb/mnist/
The Fashion MNIST dataset can be downloaded from the zalandoresearch/fashion-mnist GitHub repository: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion

