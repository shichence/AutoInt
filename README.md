# Note
  We have moved the repo to [https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec). Please check out the latest version there.

# AutoInt

This is a TenforFlow implementation of ***AutoInt*** for CTR prediction task, as described in our paper:

Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang and Jian Tang. [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf). arXiv preprint arXiv:1810.11921, 2018.

## Requirements: 
* **Tensorflow 1.4.0-rc1**
* Python 3
* CUDA 8.0+ (For GPU)

## Introduction

AutoInt：An effective and efficient algorithm to
automatically learn high-order feature interactions for (sparse) categorical and numerical features.

<div align=center>
  <img src="https://github.com/shichence/AutoInt/blob/master/figures/model.png" width = 50% height = 50% />
</div>
The illustration of AutoInt. We first project all sparse features
(both categorical and numerical features) into the low-dimensional space. Next, we feed embeddings of all fields into stacked multiple interacting layers implemented by self-attentive neural network. The output of the final interacting layer is the low-dimensional representation of learnt combinatorial features, which is further used for estimating the CTR via sigmoid function.

## Usage
### Input Format
AutoInt requires the input data in the following format:
* train_x: matrix with shape *(num_sample, num_field)*. train_x[s][t] is the feature value of feature field t of sample s in the dataset. The default value for categorical feature is 1.
* train_i: matrix with shape *(num_sample, num_field)*. train_i[s][t] is the feature index of feature field t of sample s in the dataset. The maximal value of train_i is the feature size.
* train_y: label of each sample in the dataset.

If you want to know how to preprocess the data, please refer to `./Dataprocess/Criteo/preprocess.py`

### Example
We use four public real-world datasets(Avazu, Criteo, KDD12, MovieLens-1M) in our experiments. Since the first three datasets are super huge, they can not be fit into the memory as a whole. In our implementation, we split the whole dataset into 10 parts and we use the first file as test set and the second file as valid set. We provide the codes for preprocessing these three datasets in `./Dataprocess`. If you want to reuse these codes, you should first run `preprocess.py` to generate `train_x.txt, train_i.txt, train_y.txt` as described in `Input Format`. Then you should run `./Dataprocesss/Kfold_split/StratifiedKfold.py` to split the whole dataset into ten folds. Finally you can run `scale.py` to scale the numerical value(optional).

To help test the correctness of the code and familarize yourself with the code, we upload the first `10000` samples of `Criteo` dataset in `train_examples.txt`. And we provide the scripts for preprocessing and training.(Please refer to `	sample_preprocess.sh` and `test_code.sh`, you may need to modify the path in `config.py` and `test_code.sh`). 

After you run the `test_code.sh`, you should get a folder named `Criteo` which contains `part*, feature_size.npy, fold_index.npy, train_*.txt`. `feature_size.npy` contains the number of total features which will be used to initialize the model. `train_*.txt` is the whole dataset. If you use other small dataset, say `MovieLens-1M`, you only need to modify the function `_run_` in `train.py`.

Here's how to run the preprocessing.
```
mkdir Criteo
python ./Dataprocess/Criteo/preprocess.py
python ./Dataprocess/Kfold_split/stratifiedKfold.py
python ./Dataprocess/Criteo/scale.py
```

Here's how to run the training.
```
python -u train.py \
                       --data "Criteo"  --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save "True" --save_path "./test_code/Criteo/b3h2_64x64x64/"  \
                       --field_size 39  --run_times 1 --data_path "./" \
                       --epoch 3 --has_residual "True"  --has_wide "False" \
                       --batch_size 1024 \
                       > test_code_single.out &
```

You should see output like this:

```
...
train logs
...
start testing!...
restored from ./test_code/Criteo/b3h2_dnn_dropkeep1_400x2/1/
test-result = 0.8088, test-logloss = 0.4430
test_auc [0.8088305055534442]
test_log_loss [0.44297631300399626]
avg_auc 0.8088305055534442
avg_log_loss 0.44297631300399626
```

## Citation
If you find AutoInt useful for your research, please consider citing the following paper:
```
@article{weiping2018autoint,
  title={AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks},
  author={Weiping, Song and Chence, Shi and Zhiping, Xiao and Zhijian, Duan and Yewen, Xu and Ming, Zhang and Jian, Tang},
  journal={arXiv preprint arXiv:1810.11921},
  year={2018}
}
```


## Contact information
If you have questions related to the code, feel free to contact Weiping Song (`songweiping@pku.edu.cn`), Chence Shi (`chenceshi@pku.edu.cn`) and Zhijian Duan (`zjduan@pku.edu.cn`).

## License
MIT

## Acknowledgement
This implementation gets inspirations from Kyubyong Park's [transformer](https://github.com/Kyubyong/transformer) and Chenglong Chen' [DeepFM](https://github.com/ChenglongChen/tensorflow-DeepFM).
