# Generative-Adversarial-Networks
去年做的基于MxNet/Gluon实现GAN

## 关于使用网络模型
当前有两种网络，一种全连接模型，另外一种是CNN模型，对于模型代码的实现另外建立了一个.py文件，你可以根据自己的需要在main.py中import自己想要模型对应的.py文件，或者自己选择另外建立一个.py文件实现自己的模型
* GAN.py是全连接模型
* DCGAN.py是CNN模型

## 关于训练参数
训练参数在main.py中调整，具体请看main.py

## 关于训练
一切准备就绪后，选择main.py运行python程序即可，中间输出的图片可在sample目录下查看，模型参数则保存在model目录中，若要重新加载训练参数，可以在GAN.py或者DCGAN.py中调用load_param函数

## 关于样本
程序默认使用MNIST数据集，您可以根据自己的需要使用.rec文件或者image文件
* .rec文件请调用dataOp.py文件中的load_data_from_rec函数
* image文件请调用dataOp.py文件中的load_data_from_img函数
