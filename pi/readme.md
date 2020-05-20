# 树莓派上编译安装pytorch

## 为什么要在树莓派上安装pytorch
树莓派是一个香烟盒大小的电脑，能运行window（IOT）和linux系统。可以当做一台普通的电脑用来办公上网，还有裸露的针脚可以用来控制你自己设计的电路。比如读取各种（温度，重力，加速度）传感器信息，也可以驱动马达和蜂鸣器，摄像头什么的。

说到可以驱动摄像头，那么我们就可以通过pytorch进行推理，树莓派的配置很低，毕竟几百块钱的成本，配置不会高到哪里去，但是通过他的cpu还是能够处理一些简单的推理工作的。


## 系统环境安装

这部分就略掉了，主要就是要将系统安装到SD卡中并插入树莓派，这个官网都有介绍，就不细说了。
我这里使用的是树莓派4b:1.5GHz四核64位ARM Cortex-A72 CPU， 4G内存
系统也是官官方的基于Debian 10 Buster的ARM版linux.

进入系统后，首先还是安装conda，因为arm版的conda很久都不更新了，所以有个大佬专门制作了树莓派版的conda ： https://github.com/jjhelmus/berryconda 我们按照说明直接安装后就可以使用了

因为pytorch没有官方的arm版提供下载，所以我们需要在树莓派上自己进行编译

下面安装编译pytorch所需要的包
```bash
sudo apt install libopenblas-dev libblas-dev m4 cmake cython
```
继续安装python的包
```bash
pip install numpy pyyaml cyphon
```
这里如果不安装numpy的话也能成功编译，但是编译出来的PyTorch不支持numpy

## 编译pytorch 1.4

做新版就是1.4 所以我们这里拿最新版来做

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.4 #这里选择最新的1.4版
git submodule sync
git submodule update --init --recursive
git submodule update --remote third_party/protobuf #这句必须要有，否则在编译时会报一个找不到protobuf.h的错误
```
树莓派是不支持CUDA和MKLDNN的，CUDA是nv的，MKLDNN是intel的，
我们拿树莓派也只做推理，分布式也不要了。
所以我们设置以下的环境变量
```bash
export NO_CUDA=1
export NO_DISTRIBUTED=1
export NO_MKLDNN=1
export MAX_JOBS=4 #这里设置4是因为4b是4核，如果树莓派是3的话，设置成1
```

进行完以上的配置，我们可以编译了

```bash
#本地安装
python setup.py install

#打包成whl，打包成功后这个文件在dist目录里面
python setup.py bdist_wheel
```

编译时一个漫长的过程，我的4b上大概花了2个半小时。听说3需要5个小时左右。慢慢等，不要着急，按照我上面的步骤肯定是可以成功的。

## 安装 torchvision 
编译完pytorch以后我们肯定还需要安装torchversion，pip的arm源里面torchvision的版本是0.22，已经是一个很老的版本了，所以这里面我们还是通过源码自己编译。

安装编译所需要的包,这里主要是编译pillow使用的，因为torchvision是基于
```bash
sudo apt-get install libjpeg-dev libavcodec-dev libavformat-dev libswscale-dev
```
安装pillow
```
pip install pillow
```
编译 torchvision

目前官网的torchvision版本是0.6，我们不用切换版本直接用就好了
```bash
git clone https://github.com/pytorch/vision.git

cd vision
git checkout v0.5 #如果需要使用与pytorch 1.4一同发布的0.5版，则要加上这句
#本地安装
python setup.py install

#打包成whl
python setup.py bdist_wheel
```

等待完成，就可以使用了



## 下载
为了节省大家的时间，我这里也将我编译好的包提供给大家下载

我这个是基于python 3.6 进行编译的，如果大家用3.6的话，直接下载安装即可

[pytorch 1.4](torch-1.4.0a0+7963631-cp36-cp36m-linux_armv7l.whl)

[torchvision 0.6](torchvision-0.6.0a0+bb5af1d-cp36-cp36m-linux_armv7l.whl)



## 参考

这边日本人写的文章里面 编译遇到的问题总结的比较全：
https://qiita.com/yyojiro/items/d91b02149aa6480ded80


这里只有 1.2以前版本的，想装以前版本可以直接从这里下：
https://github.com/nmilosev/pytorch-arm-builds