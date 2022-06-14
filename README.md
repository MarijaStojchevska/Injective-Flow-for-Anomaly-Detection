```
Usage: train.py [--path PATH] [--dataset DATASET]
                [--testset_type TESTSET_TYPE] [--categoryName CLASSNAME]
                [--featureExtractor FEATUREEXTRACTOR] [--runTrain RUNTRAIN]
                [--num_epochs NUM_EPOCHS] [--threshold THRESHOLD]
                [--batch_size BATCH_SIZE] [--lr LR]

Arguments:
--path             path to the main directory
--dataset          which dataset to work with [MNIST, MVTEC]
--testset_type     which MNIST testset to work with [diagonal, off-diagonal, spots, cross, mixed, inverted, fashion]
--categoryName     which MVTEC category to work with [bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper]
--featureExtractor which feature extractor to use [densenet, vgg16]
--runTrain         whether to run the training procedure
--num_epochs       number of training epochs
--threshold        when should the ml training begin
--batch_size       size of the training batch
--lr               learning rate



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
from the MVTec dataset. In other words, we used the injective flow to create a distribution
of healthy, non-defective images and estimate the exact likelihood of new images based on
which we decide if the new image is a part of the generated distribution (non-defective) or
if it is an outlier (defective).
</div>

<p><div align="center"><img width="567" src="https://user-images.githubusercontent.com/18449614/173446053-a69490f8-ecce-4f7f-99d0-55218c3bd9d9.png"> </div><div align="center"><i>Figure 1: MVTec dataset - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger.
The mvtec anomaly detection dataset: a comprehensive real-world dataset for unsupervised
anomaly detection. International Journal of Computer Vision, 129(4):1038–1059,
2021.</i></div></p>


<p><div align="justify">
The architecture of the injective flows allows for efficient likelihood computation of a new
sample with respect to two different densities learned in the two different output spaces of
the model: the output space of the bijective map, and the output space of the injective map. 
Therefore, after training the model on nonanomalous examples, we evaluated
the likelihoods—the probability that the instance is part of the learned distribution—for
each test instance relative to the two learned densities. Our goal was to check if these two
likelihoods are drastically different. Intuitively speaking, we were checking if something that
looks like an anomaly in one space looks nonanomalous in the other space, and vice versa.
From the evaluation results summarized in an AUC-ROC curve, we observed that, for the
purpose of outlier detection, the difference in densities in both spaces is not large and can be
neglected. This indicates that, as we intuitively expected, the injective mapping contributes
to a faster but not a better model evaluation. Hence, for problems involving the use of these
models in learning low-dimensional data distribution, we can neglect injective mapping and
rely solely on traditional normalizing flows. </div></p>

<div align="justify">
Given the complexity of our work, at the very beginning, we facilitated a thorough evaluation
of the generative and discriminatory power of the model by using the MNIST dataset that is
suitable for deep learning. By experimenting on the MNIST dataset, we concluded that the
model is capable of reconstructing high-quality images and generating new images from the
learned distribution. 

<p><div align="center"><img width="767" src="https://user-images.githubusercontent.com/18449614/173459470-8aee42ef-0d01-474d-ba2a-f573ffef3900.png"> </div><div align="center"><i>Figure 2: Example of the reconstruction of 36 input MNIST images using injective models
trained on 30,000 MNIST training examples. For each model, the different depth of the
injective map, i.e. the number of squeeze-bijective revnet-injective revnet blocks, is written
above each column. The first row shows the reconstructions of the whole injective-bijective
transformation, while the second row shows their corresponding reconstructions obtained
by applying only the inverse bijective transformation.</i></div></p>

<p><div align="center"><img width="500" src="https://user-images.githubusercontent.com/18449614/173459774-567f1050-0977-452d-92ee-28b2e1af9f12.png"> </div><div align="center"><i>Figure 3: Example of newly generated digits using an injective model trained on 30,000
MNIST training images. Above each image we show the depth of the injective map of the
model that generates the displayed digits.</i></div></p>
  
  
In contrast, we also noticed that the injective flow with deeper injective
mappings becomes quite unstable in reconstructing outliers. Furthermore, we tested the
discriminatory performance in anomaly detection of the model based on the MNIST dataset
using seven different test sets, of which six were artificially created. Having concluded that
the model has a remarkable ability to detect anomalies for handwritten digits, we proceeded
to work on the same problem for the MVTec dataset. </div>

<p><div align="center"><img width="567" src="https://user-images.githubusercontent.com/18449614/173460172-20a1cf45-ed1d-4e42-bbfe-9934354ceec1.png"> </div><div align="center"><i>Figure 4: Example of out-of-distribution data reconstruction using an injective model
trained on the MNIST dataset.</i></div></p>

<p><div align="justify">
Given the dimensionality of the MVTec images, we used VGG16 and DenseNet-121 transfer learning to extract their
features and thus reduce their dimension, while still preserving important image information.
The extracted features were then averaged across the channel dimension. Apart from
the injective mapping, in this way we further reduced the computational complexity of the
model. In addition to feature extraction, we applied background extraction, data augmentation, 
and data standardization to the MVTec training and test examples. With each of
these preprocessing steps, we contributed in a different way to improving the model’s performance.
In the rest of the work, we covered the training and evaluation of the model on
such preprocessed images. </div></p>

<p><div align="center"><img width="500" src="https://user-images.githubusercontent.com/18449614/173462339-06aee483-12a3-4e18-8c23-753e71b08b9d.png"> </div><div align="center"><i>Table 1: Evaluation results of the injective model when trained and tested on the VGG16
extracted features per category. The results are shown with respect to the original, standardized,
and augmented input images. The upper values in each row represent the AUC
of the performance of the injective flow, while the lower values represent the AUC of the
performance of the bijective flow. The best value for each category is shown in bold.</i></div></p>

<p><div align="center"><img width="667" src="https://user-images.githubusercontent.com/18449614/173462473-b921ec77-be0b-46c0-9b4a-79c63e4b285b.png"> </div><div align="center"><i>Table 2: Evaluation results of the injective model when trained and tested on the
DenseNet-121 extracted features per category. The results are shown with respect to the
original, standardized, and two differently augmented datasets. The upper values in each
row represent the AUC of the injective flow, while the lower values represent the AUC of
the bijective flow. The best value for each category is shown in bold. </i></div></p>
  
As a final thought, we can point out that our
injective model outperforms the established baselines in detecting MVTec defective objects
and textures for most of the categories.

<p><div align="center"><img width="500" src="https://user-images.githubusercontent.com/18449614/173462817-fd9dbf24-977d-461b-94de-cb8a4f0c351a.png" > </div><div align="center"><i>Table 3: Comparison of the best AUC values obtained for the injective model relative to
those corresponding to the baseline models. The best results for each MVTec category
are shown in bold. </i></div></p>
