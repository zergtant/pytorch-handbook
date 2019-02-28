# PyTorch 中文手册（pytorch handbook）
![pytorch](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)

## 书籍介绍
这是一本开源的书籍，目标是帮助那些希望和使用PyTorch进行深度学习开发和研究的朋友快速入门。

由于本人水平有限，在写此教程的时候参考了一些网上的资料，在这里对他们表示敬意，我会在每个引用中附上原文地址，方便大家参考。

深度学习的技术在飞速的发展，同时PyTorch也在不断更新，且本人会逐步完善相关内容。

## 版本说明
由于PyTorch版本更迭，教程的版本会与PyTorch版本，保持一致。

12月8日PyTorch已经发布1.0的稳定版。
API的改动不是很大，本教程已经通过测试，保证能够在1.0中正常运行。
不过目前看影响不大，因为毕竟内容还不多。 v0.4.1已经新建了分支作为存档，并且该不会再进行更新了。

目前最近版本为1.0.1，没有大的API变更，所以还是以大版本1.0作为开发版本

[官方1.0说明](https://github.com/pytorch/pytorch/releases/tag/v1.0.0)
[主要改动中文说明](changelog-v1.0.md)

## QQ群
群号：985896536

扫描二维码

![QR](PyTorch-Handbook-QR.png)


或

[点击链接加入群聊 『PyTorch Handbook 交流群』](https://jq.qq.com/?_wv=1027&k=5L28MSZ)

## 说明

修改错别字请直接提issue或PR

PR时请注意版本

有问题也请直接提issue

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
4. [数据的加载和预处理](chapter2/2.1.4-pytorch-basics-data-lorder.ipynb)
#### 第二节 深度学习基础及数学原理

[深度学习基础及数学原理](chapter2/2.2-deep-learning-basic-mathematics.ipynb)

#### 第三节 神经网络简介

[神经网络简介](chapter2/2.3-deep-learning-neural-network-introduction.ipynb)

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

[可视化理解卷积神经网络](chapter4/4.2.2-tensorboardx.ipynb)

#### 第三节 fastai
#### 第四节 训练的一些技巧
#### 第五节 并行计算

### 第五章 应用
#### 第一节 Kaggle介绍
#### 第二节 结构化数据
#### 第三节 计算机视觉
#### 第四节 自然语言处理
#### 第五节 协同过滤

### 第六章 资源


### 第七章 附录
transforms的常用操作总结

pytorch的损失函数总结

pytorch的优化器总结

## License

![](https://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png)

[本作品采用知识共享署名-非商业性使用-相同方式共享 3.0  中国大陆许可协议进行许可](http://creativecommons.org/licenses/by-nc-sa/3.0/cn)
