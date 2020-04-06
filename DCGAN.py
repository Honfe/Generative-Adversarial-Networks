from mxnet import gluon
from mxnet import ndarray as nd
import mxnet as mx

class Reshape(gluon.nn.Block):
    _deep = None
    _high = None
    _width = None

    def __init__(self, deep=1, high=1, weight=1, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self._deep = deep
        self._high = high
        self._width = weight

    def forward(self, x):
        return nd.reshape(x, shape=(-1, self._deep, self._high, self._width))

# 生成器网络
def g_net(ctx=mx.cpu()):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(3*3*256, activation='relu'))
        net.add(Reshape(256, 3, 3))
        net.add(gluon.nn.Conv2DTranspose(channels=256, kernel_size=3,
                                         strides=(2, 2), padding=(0, 0)))
        net.add(gluon.nn.BatchNorm(axis=1))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Conv2DTranspose(channels=128, kernel_size=4,
                                         strides=(2, 2), padding=(1, 1)))
        net.add(gluon.nn.BatchNorm(axis=1))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Conv2DTranspose(channels=1, kernel_size=4,
                                         strides=(2, 2), padding=(1, 1), activation='sigmoid'))
    net.initialize(ctx=ctx)
    return net

# 判别器网络
def d_net(ctx=mx.cpu()):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(
            Reshape(1, 28, 28),
            gluon.nn.Conv2D(channels=128, kernel_size=4, strides=(2, 2), padding=(1, 1)),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            # gluon.nn.MaxPool2D(pool_size=2, strides=2),
            gluon.nn.Conv2D(channels=256, kernel_size=4, strides=(2, 2), padding=(1, 1)),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            # gluon.nn.MaxPool2D(pool_size=2, strides=2),
            gluon.nn.Flatten(),
            # gluon.nn.Dense(128, activation='relu'),
            gluon.nn.Dense(1, activation='sigmoid')
        )
    net.initialize(ctx=ctx)
    return net

# 生成器损失函数
def loss_g(D_fake):
    G_loss = -nd.mean(nd.log(D_fake))
    return G_loss

# 判别器损失函数
def loss_d(D_real, D_fake):
    D_loss = -nd.mean(nd.log(D_real) + nd.log(1. - D_fake))
    return D_loss

# 保存模型参数
def save_param(net, path):
    net.save_paramters(path)

# 加载模型
def load_param(net, path):
    net.load_paramters(path)
