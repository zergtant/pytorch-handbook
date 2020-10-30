# pytorch 大版本的主要更新总结


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

