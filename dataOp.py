from mxnet import gluon

def transform(data, label):
    return data.astype('float32') / 255., label.astype('float32') / 255.

# 读取MNIST数据集（自动从网络读取）
def load_data_from_mnist(batch_size):
    mnist = gluon.data.vision.MNIST(train=True, transform=transform)
    image = gluon.data.DataLoader(mnist, batch_size, shuffle=False, last_batch='discard')
    return image

# 读取本地.rec数据集
def load_data_from_rec(rec_path, batch_size, pic_size):
    train = gluon.data.vision.ImageRecordDataset(rec_path, flag=1)
    transform_tools = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.Resize(pic_size),
        gluon.data.vision.transforms.ToTensor()
    ])
    train = train.transform_first(transform_tools)
    image = gluon.data.DataLoader(train, batch_size, shuffle=False, last_batch='discard')
    return image

# 读取本地图片集
def load_data_from_img(img_path, batch_size, pic_size):
    train = gluon.data.vision.ImageFolderDataset(img_path)
    transform_tools = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.Resize(pic_size),
        gluon.data.vision.transforms.ToTensor()
    ])
    train = train.transform_first(transform_tools)
    image = gluon.data.DataLoader(train, batch_size, shuffle=False, last_batch='discard')
    return image
