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

<div align=center>
  <img src="https://github.com/shichence/AutoInt/blob/master/figures/model.png" width = 50% height = 50% />
</div>
The illustration of AutoInt. We first projects all sparse features
(both categorical and numerical features) into the low-dimensional space. Next, we feed embeddings of all fields into a interacting layer implemented by self-attentive neural network. The output of the final interacting layer is the low-dimensional representation of the input feature, which is further used for estimating the CTR via sigmoid function.

## Usage
### Input Format
The implementation requires the input data in the following format:
* train_x: matrix with shape *(num_sample, num_field)*. train_x[s][t] is the feature value of feature field t of sample s in the dataset. The default value for categorical feature is 1.
* train_i: matrix with shape *(num_sample, num_field)*. train_i[s][t] is the feature index of feature field t of sample s in the dataset. The maximal value of train_i is the feature size.
* train_y: label of each sample in the dataset.

If you want to know how to preprocess the data, please refer to `./Dataprocess/Criteo/preprocess.py`

### 



## Acknowledgement
This code is based on the [previous work by Kyubyong](https://github.com/Kyubyong/transformer). Many thanks to [Kyubyong Park](https://github.com/Kyubyong).
