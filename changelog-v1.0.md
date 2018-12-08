# PyTorch 1.0主要改动说明
本改动说明只针对主要部分进行阐述，详情还请以官方英文为准

## JIT
JIT是一组编译工具，用来弥补研究和产品部署之间的差距。

新增的torch.jit包，包含了一组编译器工具，用于搭建PyTorch的研究与工业生产之间的桥梁。
它包括一种名为Torch Script的语言(单从语法上来看，这是Python语言的一个子集，所以我们不用再学一门新语言了)，以及两种使现有代码与JIT兼容的方法。
Torch Script的代码是可以进行积极的优化（类似TensorFlow代码），可以序列化，以供以后在C++ API中使用，它完全不依赖于Python。

`@torch.jit.script`注解，官方给出的注释是Write in Python, run anywhere!

## 改进的分布式库torch.distributed
- 新的torch.distributed是性能驱动的，并且对所有后端（Gloo，NCCL和MPI）完全异步操作
- 显著的分布式数据并行性能改进，尤其适用于网络较慢的主机，如基于以太网的主机
- 为torch.distributed包中的所有分布式集合操作添加异步支持
- 在Gloo后端添加以下CPU操作：send，recv，reduce，all_gather，gather，scatter
- 在NCCL后端添加障碍操作
- 为NCCL后端添加new_group支持

其实主要含义就是分布式的性能得到了改进，具体改进多少后面还要进行测试了

##  C++的前端（C++ 版 Torch)
C++前端是到PyTorch后端的纯C++接口，遵循已建立的Python前端的API和体系结构。它旨在支持在高性能、低延迟和硬核C++应用程序中的研究。它相当于torch.nn, torch.data等PyTorch的Python前端组件。

这个对我来说意义不大，不详细说了

## Torch Hub
这个一听就知道，google刚推出TensorFlow Hub不久，FB就跟进了，受益最大的就是“没有技术含量的公司可以再吹一波牛逼了”

## 官方提供免费课程
与Udacity合作，免费提供成人AI课程：

[PyTorch深度学习简介](https://cn.udacity.com/course/deep-learning-pytorch--ud188)

看样子如果出中文版的话可能要有一阵子了，不过 I love study，study makes me happy.


## 比较重要的改动

- torch.distributed的 TCP 后端被移除了，官方建议：CPU用Gloo或者MPI, GPU用英伟达的NCCL
- 使用tensor的0下标返回数据的方法（如：loss[0]），彻底移除, 你需要使用loss.item() 这个可能比较重要，因为标量（0阶张量）的操作一定要用到这个
- 移除了直接在CUDA上调用numpy函数的隐式类型转换，现在你需要手动把数据从CUDA先移到CPU上 这个对我们来说不重要，我们一直都是先移动到cpu再转化成numpy函数的