```
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
- The whole MVTec AD dataset can be downloaded from the official MVTec website: https://www.mvtec.com/company/research/datasets/mvtec-ad
- The official link for downloading the MNIST dataset: http://yann.lecun.com/exdb/mnist/
- The Fashion MNIST dataset can be downloaded from the zalandoresearch/fashion-mnist GitHub repository: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
```

# Detection of Anomalous Images using Injective Flows
<div align="justify">
Traditional normalizing flows require large computational costs to learn transformations of
an input distribution, mainly because they operate at exactly the same dimension as the
input which is usually high dimensional. We utilized the Trumpet model idea (https://github.com/swing-research/trumpets.git) to implement
an injective flow capable of mitigating the computational complexity in normalizing flows
via an injective mapping. The main task to which we adapted this model is the detection of
defects in the manufacturing industry by working with images of various objects and textures
from the MVTec dataset:
</div>

<div align="center"  padding="50px" ><img width="567" alt="Screenshot 2022-06-13 at 23 10 22" src="https://user-images.githubusercontent.com/18449614/173446053-a69490f8-ecce-4f7f-99d0-55218c3bd9d9.png"> </div>

In other words, we used the injective flow to create a distribution
of healthy, non-defective images and estimate the exact likelihood of new images based on
which we decide if the new image is a part of the generated distribution (non-defective) or
if it is an outlier (defective).
  
  
  

