from matplotlib import pyplot as plt

# 图片展示
def show_picture(sample, epoch):
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i * 4 + j + 1)
            plt.imshow((sample[i * 4 + j].reshape((28, 28)).asnumpy()), cmap='Greys')
    plt.savefig('./sample/' + str(epoch) + '.jpg')