# PyTorch 中文手册（pytorch handbook）
![pytorch](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)

## 书籍介绍
这是一本开源的书籍，目标是帮助那些希望和使用PyTorch进行深度学习开发和研究的朋友快速入门。

由于本人水平有限，在写此教程的时候参考了一些网上的资料，在这里对他们表示敬意，我会在每个引用中附上原文地址，方便大家参考。

深度学习的技术在飞速的发展，同时PyTorch也在不断更新，且本人会逐步完善相关内容。

## 版本说明
由于PyTorch版本更迭，教程的版本会与PyTorch版本，保持一致。

[pytorch大版本更新的主要变动总结](pytorch-changelog.md)  当前版本 1.7

## QQ 5群 

群号：894059877

扫描二维码

![QR](Pytorch-Handbook-5.png) 

[点击链接加入群聊 『PyTorch Handbook 交流5群』](//jq.qq.com/?_wv=1027&k=HI1gtcVa)

1群(985896536)已满，2群(681980831) 3群(773681699)已满  4群(884017356)已满 

不要再加了

## 新福利

公众账号每日分享干货文章
![weixin QR](deephub.jpg) 



## 说明

- 修改错别字请直接提issue或PR

- PR时请注意版本

- 有问题也请直接提issue

感谢

## 目录

### 第一章：PyTorch 入门

1. [PyTorch 简介](chapter1/1.1-pytorch-introduction.md)
2. [PyTorch 环境搭建](chapter1/1.2-pytorch-installation.md)
3. [PyTorch 深度学习：60分钟快速入门（官方）](chapter1/1.3-deep-learning-with-pytorch-60-minute-blitz.md)
    - [张量](chapter1/1_tensor_tutorial.ipynb)
    - [Autograd：自动求导](chapter1/2_autograd_tutorial.ipynb) 
    - [神经网络](chapter1/3_neural_networks_tutorial.ipynb)
    - [训练一个分类器](chapter1/4_cifar10_tutorial.ipynb)
    - [选读：数据并行处理（多GPU）](chapter1/5_data_parallel_tutorial.ipynb)
4. [相关资源介绍](chapter1/1.4-pytorch-resource.md)

### 第二章 基础
#### 第一节 PyTorch 基础
1. [张量](chapter2/2.1.1.pytorch-basics-tensor.ipynb)
2. [自动求导](chapter2/2.1.2-pytorch-basics-autograd.ipynb)
3. [神经网络包nn和优化器optm](chapter2/2.1.3-pytorch-basics-nerual-network.ipynb)
4. [数据的加载和预处理](chapter2/2.1.4-pytorch-basics-data-loader.ipynb)
#### 第二节 深度学习基础及数学原理

[深度学习基础及数学原理](chapter2/2.2-deep-learning-basic-mathematics.ipynb)

#### 第三节 神经网络简介

[神经网络简介](chapter2/2.3-deep-learning-neural-network-introduction.ipynb)  注：本章在本地使用微软的Edge打开会崩溃，请使Chrome Firefox打开查看

#### 第四节 卷积神经网络

[卷积神经网络](chapter2/2.4-cnn.ipynb)

#### 第五节 循环神经网络

[循环神经网络](chapter2/2.5-rnn.ipynb)

### 第三章 实践
#### 第一节 logistic回归二元分类

[logistic回归二元分类](chapter3/3.1-logistic-regression.ipynb)


#### 第二节 CNN:MNIST数据集手写数字识别

[CNN:MNIST数据集手写数字识别](chapter3/3.2-mnist.ipynb)

#### 第三节 RNN实例：通过Sin预测Cos

[RNN实例：通过Sin预测Cos](chapter3/3.3-rnn.ipynb)

### 第四章 提高
#### 第一节 Fine-tuning

[Fine-tuning](chapter4/4.1-fine-tuning.ipynb)

#### 第二节 可视化

[visdom](chapter4/4.2.1-visdom.ipynb)

[tensorboardx](chapter4/4.2.2-tensorboardx.ipynb) 

[可视化理解卷积神经网络](chapter4/4.2.3-cnn-visualizing.ipynb)

#### 第三节 Fast.ai
[Fast.ai](chapter4/4.3-fastai.ipynb)
#### 第四节 训练的一些技巧

#### 第五节 多GPU并行训练
[多GPU并行计算](chapter4/4.5-multiply-gpu-parallel-training.ipynb)

#### 补充翻译文章：在PyTorch中使用DistributedDataParallel进行多GPU分布式模型训练
[在PyTorch中使用DistributedDataParallel进行多GPU分布式模型训练](chapter4/distributeddataparallel)


### 第五章 应用
#### 第一节 Kaggle介绍
[Kaggle介绍](chapter5/5.1-kaggle.md)
#### 第二节 结构化数据
[Pytorch处理结构化数据](chapter5/5.2-Structured-Data.ipynb)
#### 第三节 计算机视觉
[Fashion MNIST 图像分类](chapter5/5.3-Fashion-MNIST.ipynb)
#### 第四节 自然语言处理
#### 第五节 协同过滤

### 第六章 资源


### 第七章 附录

[树莓派编译安装 pytorch 1.4](pi/)

transforms的常用操作总结

pytorch的损失函数总结

pytorch的优化器总结

## License

![](https://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png)

[本作品采用知识共享署名-非商业性使用-相同方式共享 3.0  中国大陆许可协议进行许可](http://creativecommons.org/licenses/by-nc-sa/3.0/cn)
