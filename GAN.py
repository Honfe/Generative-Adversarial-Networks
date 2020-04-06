from mxnet import gluon
from mxnet import ndarray as nd
import mxnet as mx

# 生成器网络
def g_net(ctx=mx.cpu()):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(256, activation='relu'))
        net.add(gluon.nn.Dense(784, activation='sigmoid'))
    net.initialize(ctx=ctx)
    return net

# 判别器网络
def d_net(ctx=mx.cpu()):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(256, activation='relu'))
        net.add(gluon.nn.Dense(1, activation='sigmoid'))
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

# 保存模型
def save_param(net, path):
    net.save_paramters(path)

# 加载模型
def load_param(net, path):
    net.load_paramters(path)
