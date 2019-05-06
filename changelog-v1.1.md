# PyTorch 1.1主要改动说明
本改动说明只针对主要部分进行阐述，详情还请以官方英文为准 [官网地址](https://github.com/pytorch/pytorch/releases/tag/v1.1.0)

**重要 ： CUDA 8.0不再被支持了**

## TensorBoard 
TensorBoard 已经被官方支持了（实验中）

`from torch.utils.tensorboard import SummaryWriter`
使用这个语句可以直接引用

## DistributedDataParallel 新的功能额
`nn.parallel.DistributedDataParallel`: 现在可以包装多GPU模块，它可以在一台服务器上实现模型并行和跨服务器的数据并行等用例。

## 一些主要更新
- TorchScript(Pytorch JIT)更快、更好的支持自定义RNN

- 可以在ScriptModule上通过使用torch.jit包装属性来分配属性
-  TorchScript现在对列表和字典类型提供了鲁棒性的支

- 对于更复杂的有状态操作，TorchScript现在支持使用`@torch.jit.script`注释类
