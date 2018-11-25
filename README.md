# AutoInt

Code for the paper [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf).

## Requirements: 
* **Tensorflow 1.4.0-rc1**
* Python 3
* CUDA 8.0+ (For GPU)

## Introduction

AutoInt：An effective and efficient algorithm to
automatically learn the high-order feature combinations of input
features

![]('/Users/chenceshi/Desktop/AutoInt/figures/model.png')

The illustration of AutoInt. We first projects all sparse features
(both categorical and numerical features) into the low-dimensional space. Next, we feed embeddings of all fields into a interacting layer implemented by self-attentive neural network. The output of the final interacting layer is the low-dimensional representation of the input feature, which is further used for estimating the CTR via sigmoid function.



## acknowledgement
This code is based on the [previous work by Kyubyong](https://github.com/Kyubyong/transformer). Many thanks to [Kyubyong Park](https://github.com/Kyubyong).
