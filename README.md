# DeepPyramid DPM

### Introduction

The DeepPyramid DPM repository is code for our CVPR 2015 paper
[Deformable Part Models are Convolutional Neural Networks](http://www.cs.berkeley.edu/~rbg/papers/cvpr15/dpdpm.pdf).
This work was completed while Ross Girshick was a postdoc at UC Berkeley (prior
to joining Microsoft Research).

    @inproceedings{girshick2015dpdpm,
        Author    = {Ross Girshick and Forrest Iandola and
                     Trevor Darrell and Jitendra Malik},
        Title     = {Deformable Part Models are
                     Convolutional Neural Networks},
        Booktitle = {Proceedings of the IEEE Conference on
                     Computer Vision and Pattern Recognition ({CVPR})},
        Year      = {2015}
    }

It implements Latent SVM training of deformable part models (DPMs) on top of deep feature pyramids.

### License

Deep Pyramid DPM is released under the simplified BSD license (refer to the LICENSE file for details).

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/dp-dpm-data/Makefile.config) for reference.
2. MATLAB (tested on R2014a)

### Requirements: hardware

A good GPU, such as a GTX Titan, K20, etc.

### Installation

1. Clone the repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/rbgirshick/dp-dpm.git
  ```

2. We'll call the directory that you cloned DeepPyramid DPM into `DPDPM_ROOT`

3. Build Caffe and matcaffe
    ```Shell
    cd $DPDPM_ROOT/caffe
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make matcaffe
    # Replace 8 with your favorite number of compile threads
    ```

4. Download pre-computed ImageNet model
    ```Shell
    cd $DPDPM_ROOT
    ./data/scripts/fetch_imagenet_model.sh
    ```

    This will populate the `$DPDPM_ROOT/data/caffe_nets` folder with `CaffeNet.v2.caffemodel`.

5. Symlink to the PASCAL VOC 2007 dataset
    ```Shell
    cd $DPDPM_ROOT
    # Replace /path/to with the actual path to VOC 2007
    ln -s /path/to/VOC2007/VOCdevkit cachedir/VOC2007/VOCdevkit
    ```

6. Build the MATLAB/mex code
    ```Shell
    # Start matlab
    matlab
    >> compile
    ```

### Usage

To train and test a detector:

```Shell
cd $DPDP_ROOT
matlab
>> % run these commands inside matlab
>> precompute_feat_pyramids() % this is take several hours
>> pascal('bicycle', 3) % replace 'bicycle' with any PASCAL class
```

Executing `precompute_feat_pyramids()` is optional. If you don't call this function
then features will be cached while training and testing for the first time (making
training and testing much slower).
