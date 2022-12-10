# pytorch 大版本的主要更新总结

## 1.11 2022-3-14

更新了一个新库 TorchData 

beta 版本 functorch 可组合函数转换。

支持Python 3.10

## 1.10 2021-10-22

本版其实最大的变化是交叉熵损失增加了label_smoothing，再也不用自己实现了

```
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)
```

支持Android NNAPI （测试版）

torch.fx 已经变为稳定版，也就是说可以在生产上用了

torch.special、nn.Module 参数化（稳定版）

分布式训练的Remote Module、DDP Communication Hook、ZeroRedundancyOptimizer也都变为稳定版



## 1.9 2021-06-16
torch.linalg,Complex Autograd 已经更新为（稳定版）

torch.norm 已经弃用了，需要使用 torch.linalg.norm

TorchVision库已经可以在IOS/Android上使用

PyTorch Profiler（测试版）: 利用Kineto进行GPU分析，TensorBoard 进行可视化，PyTorch使用教程和文档都已经完善。

TorchVision 0.10：添加了新的SSD和SSDLite模型

TorchAudio 0.9.0：可在非Python 环境（包括C++、Android和iOS）中部署的wav2vec 2.0模型

TorchText 0.10.0：添加了一个新的高性能Vocab模块

## 1.8 2021-03-05
支持一部分的AMD GPU（测试版）

Complex Autograd （测试版）

提高 NCCL 稳定性，包括异步错误/超时处理，RPC 分析

torch.fft 已经更新为（稳定版），也就是可以在生产环境中使用了

新增torch.linalg，为常见的线性代数运算提供与 NumPy 类似的支持（测试版）

torch.fx （测试版）可以进行 Python 代码转换

pipeline 并行化（测试版）可将 pipeline 并行化作为训练 loop 的一部分



## 1.7 2020-10-29
支持CUDA 11：CUDA 9.2 - 11

通过 torch.fft 支持 NumPy 兼容的 FFT 操作（测试版）

Windows 系统上的分布式训练：DistributedDataParallel和集合通信提供了原型支持（原型版）

支持Nvidia A100的 原生TF32格式

PyTorch Mobile支持iOS和Android，CocoaPods和JCenter，并分别提供了二进制软件包

TORCHVISION 0.8


## 1.6 2020-7-29

官方 自动混合精度（AMP）训练 torch.amp,不需要nv的apex

torch.autograd.profiler 内存分析器 （测试版）

不支持python 3.5 以前版本

TORCHVISION 0.7

TORCHAUDIO 0.6

Pytorch1.6版本开始，PyTorch 的特性将分为 Stable（稳定版）、Beta（测试版）和 Prototype（原型版）


## 1.5 2020-4-21

C++ 前端 API（稳定型）

分布式 RPC 框架 API（稳定型）

不再支持 Python 2

TORCHVISION 0.6

## 1.4 2020-1-16

optim.lr_scheduler 持「链式更新（chaining）」。即可以定义两个 schedulers，并交替在训练中使用。

Java bindings（实验性） Java bindings 从任何 Java 程序中调用 TorchScript 模型，只支持linux

分布式模型并行训练RPC （实验性）

## 1.3 2019-10-11

命名张量（实验性） named tensor

量化支持 用 eager 模式进行 8 位模型量化

谷歌云 TPU

PyTorch Mobile 移动端 从 Python 到部署在 iOS 和安卓端

## 1.2  2019-8-9

官方实现标准的 nn.Transformer 模块

CUDA 9.2 +

TORCHVISION 0.4

D API（Domain API）：torchvision、torchtext 和 torchaudio
 
## 1.1 2019-5-1

支持 TensorBoard ： from torch.utils.tensorboard import SummaryWriter

不再支持CUDA 8.0 

TORCHVISION 0.3

## 1.0 2018-12-8

Torch Hub 预训练的模型库

JIT 编译器

C++ 前端 （实验性）

全新的分布式包torch.distributed和 torch.nn.parallel.DistributedDataParallel


## 0.4 2018-04-25

支持 Windows 系统

Tensor/Variable 合并，取消Variable

