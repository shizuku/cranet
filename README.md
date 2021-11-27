# CraNet

CraNet is a python library for Deep Learning.

## Installation

There are two ways to install CraNet

### From pip

```
pip install cranet
```

### From source

For now, This way is only available for linux.

Git clone repository

```
git clone https://github.com/shizuku/cranet.git
```

Build locally

```
cd cranet
./scripts/build_local_install.sh
```

## Basic Information

| Module | Description |
| --- | --- |
| cranet.autograd | tensor computation library based on `numpy`，集成了自动梯度的功能，支持张量的微分操作，所有需要微分的地方都用到了该库(包括梯度下降) |
| cranet.nn | 基于tensor神经网络库，支持灵活地自定义神经网络结构(自定义层数，神经元个数，计算方式) |
| cranet.optim | 优化器库，支持梯度下降神经网络参数 |
| cranet.data | 数据集库，以不同的方式处理各式各样的数据，以统一的方式将这些数据交接给神经网络，进行训练 |
| cranet.utils | 杂项，支持神经网络的储存与读入，辅助地显示一些计算过程 |

## Release

## Credits

+ [pytorch/pytorch](https://github.com/pytorch/pytorch)
+ [joelgrus/autograd](https://github.com/joelgrus/autograd)

## License
