# PosDiffNet: Positional Neural Diffusion for Point Cloud Registration in a Large Field of View

PyTorch implementation of the paper:

We exploit pytorch to achieve the PosDiffNet for point cloud registration, the datasets for training and testing are from Boreas dataset and KITTI dataset. 
These two datasets are availble at:

[Boreas] https://www.boreas.utias.utoronto.ca/
[KITTI] http://www.cvlibs.net/datasets/kitti/


The following commands are used for train and test.

# Install packages and other dependencies
python setup.py build develop

Code has been tested with Ubuntu 20.04, GCC 9.3.0, Python 3.8, PyTorch 1.7.1, CUDA 11.1 and cuDNN 8.1.0.

### Training

The code for training and testing is in `experiments/`. Use the following command for training.

CUDA_VISIBLE_DEVICES=0 python train.py


### Testing

Use the following command for testing.

CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../....pth.tar

## Acknowledgements

We thank the following repositories, which are basis for our code.

https://github.com/twitter-research/graph-neural-pde
https://github.com/rtqichen/torchdiffeq
https://github.com/qinzheng93/GeoTransformer
https://github.com/magicleap/SuperGluePretrainedNetwork
https://github.com/huggingface/transformers
https://github.com/qiaozhijian/VCR-Net
https://github.com/iMoonLab/HGNN
https://github.com/Strawberry-Eat-Mango/PCT_Pytorch
https://github.com/WangYueFt/dcp


