from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import mxnet as mx
import dataOp
# 具体使用哪一种网络，可以在此处修改
import DCGAN as net
import display

if __name__=='__main__':

    # 训练设备
    ctx = mx.gpu()

    # 训练参数
    batch_size = 128
    learning_rate = 0.0002

    # 导入数据集
    data = dataOp.load_data_from_mnist(batch_size)

    # 加载网络架构
    generator = net.g_net(ctx)
    discriminator = net.d_net(ctx)

    # 配置训练参数
    trainer_g = gluon.Trainer(generator.collect_params(), 'Adam', {'learning_rate': learning_rate})
    trainer_d = gluon.Trainer(discriminator.collect_params(), 'Adam', {'learning_rate': learning_rate})

    print('Begin to train')
    for epoch in range(151):
        loss_Dis = 0.
        loss_Gen = 0.

        for real, _ in data:

            # 加载到指定设备内存（gpu/cpu）
            real = real.as_in_context(ctx)
            Z_dim = nd.random_normal(0, 1, shape=(len(real), 100), ctx=ctx)

            # 训练判别器
            fake = generator(Z_dim)
            with autograd.record():
                real_res = discriminator(real)
                fake_res = discriminator(fake)
                loss_d = net.loss_d(real_res, fake_res)
            loss_d.backward()
            trainer_d.step(batch_size)

            # 训练生成器
            with autograd.record():
                fake = generator(Z_dim)
                fake_res = discriminator(fake)
                loss_g = net.loss_g(fake_res)
            loss_g.backward()
            trainer_g.step(batch_size)

            loss_Dis += nd.mean(loss_d).asscalar()
            loss_Gen += nd.mean(loss_g).asscalar()

        print('epoch %d: D_loss %f, G_loss %f' % (epoch, loss_Dis / len(data), loss_Gen / len(data)), ctx)

        # 保存模型并输出中间可视化结果
        if epoch % 10 == 0:
            noise = nd.random_normal(0, 1, shape=(16, 100), ctx=ctx)
            sample = generator(noise)
            net.save_param(generator, './model/')
            print('Model has been saved')
            # 输出可视化结果
            display.show_picture(sample, epoch)

    print('done!')
