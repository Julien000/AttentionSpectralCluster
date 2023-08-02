import logging
from torchvision import datasets
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from assets.datasets.load_ag import AGNEWSCustomDataset, getAgNews
# from assets.datasets.load_ag import AGNEWSCustomDataset, getAgNews
from assets.datasets.load_reuterF import GetReutersData, ReutersCustomDataset
from assets.datasets.load_sogou import GetSogoData, SoGoCustomDataset
from utils.transform import *
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy
from torchvision import transforms
# 常用的归一化方法
mean, std = 0.1307, 0.3081 #MNIST
preprocess =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)) ])

def split_train_val(trainset,valset):
    '''
     该方法用于将 训练数据分为，训练集和验证集
    '''
    labels = [trainset[i][1] for i in range(len(trainset))]
    # labels = trainset[list(range(len(trainset)))][i]
    # labels = list(map(lambda i: trainset[i][1], range(len(trainset))))
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    trainset = torch.utils.data.Subset(trainset, train_indices)
    valset = torch.utils.data.Subset(valset, valid_indices)
    return trainset, valset


# 图像数据集 ---单通道部分
#TODO: Mnist
def load_mnist(path='assets/datasets/MNIST', isVit=False):
    selfTransform = Aug_toTensor2()
    trainset = datasets.MNIST(root=path, train=True,download=True, transform=selfTransform)                
    testset = datasets.MNIST(root=path, train=False, download=True, transform=selfTransform.test_transform)
    logging.info("==== load_MNIST")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
def load_digits(path='assets/datasets', isVit=False):
    """
    DIGIST 数据集
    Digits ：共 28000 张，10 类，每一类包含相同数量数据，每一类训练集 24000 张，测试集 4000 张
    """
    selfTransform = Aug_toTensor()
    trainset = datasets.EMNIST(root=path, train=True,download=True, split="digits", transform=selfTransform)                
    testset = datasets.EMNIST(root=path, train=False, download=True,  split="digits",transform=selfTransform.test_transform)
    logging.info("==== load_Digist")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)

#TODO:EMNIST 就是MNIST 手写字母--这里用到的是Letters
def load_Letters(path='assets/datasets', isVit=False):
    """
    Letters : 共 145600 张，26 类，每一类包含相同数据，每一类训练集5600 张，测试集 800 张
    """
    selfTransform = Aug_toTensor()
    trainset = datasets.EMNIST(root=path, train=True,download=True, split="letters", transform=selfTransform)                
    testset = datasets.EMNIST(root=path, train=False, download=True,  split="letters",transform=selfTransform.test_transform)
    logging.info("==== load_Letters")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
#TODO: Qmnist
def load_qmnist(path='assets/datasets/QMNIST', isVit=False):
    selfTransform = Aug_toTensor()
    trainset = datasets.QMNIST(root=path, what='train',download=True, transform=selfTransform)                
    testset = datasets.QMNIST(root=path, what='test50k', download=True, transform=selfTransform.test_transform)
    
    logging.info("==== load_QMNIST")
    logging.info(f"Train Data: {len(trainset)}")
    
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
#TODO:fashionmnist
def load_fashionMinst(path="assets/datasets/FashionMNIST", isVit=False):

    selfTransform = Aug_toTensor()
    trainset = datasets.FashionMNIST(root=path, train=True,download=True, transform=selfTransform)                
    testset = datasets.FashionMNIST(root=path, train=False, download=True, transform=selfTransform.test_transform)
    
    logging.info("==== load_FashionMNIST")
    logging.info(f"Train Data: {len(trainset)}")
    
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
#TODO: USPS
def load_USPS(path="assets/datasets/USPS", isVit=False):
    '''
    USPS--单通道图像嘛
    '''
    selfTransform = Aug_toTensor()

    trainset = datasets.USPS(root=path, train=True,download=True, transform=selfTransform)                
    testset = datasets.USPS(root=path, train=False, download=True, transform=selfTransform.test_transform)
 
    logging.info("==== load_USPS")
    logging.info(f"Train Data: {len(trainset)}")
    
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
#TODO: FER2013

  
#CIFAR 10 RGB图像。
# 图像数据集 ---多通道部分
def load_cifar10(path='assets/datasets/CIFAR10', isVit=False):
    ## Training settings
    if isVit:
        selfTransform = Aug_toTensor()
    else:
        selfTransform = Aug_RGBtoGray()
    trainset = datasets.CIFAR10(root=path, train=True,download=True, transform=selfTransform)               
    testset = datasets.CIFAR10(root=path, train=False, download=True, transform=selfTransform.test_transform)
    # 将训练集分为训练和测试集
    logging.info("======= load_cifar10")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
#CIFAR100
def load_cifar100(path='assets/datasets/cifar100', isVit=False):
    ## Training settings
    if isVit:
        selfTransform = Aug_toTensor()
    else:
        selfTransform = Aug_RGBtoGray()
    trainset = datasets.CIFAR100(root=path, train=True,download=True, transform=selfTransform)               
    testset = datasets.CIFAR100(root=path, train=False, download=True, transform=selfTransform.test_transform)
    logging.info("==== load_CIFAR100")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)

def load_stl10(path="assets/datasets/STL10", isVit=False):
    '''
    roo图像从ImageNet中获取
    包含10个类别（飞机、鸟、汽车、猫、狗、马、猴子、船、卡车）
    每个类有标签训练样本数目为500
    无标签训练样本总数目为100000，除了含有以上提到的十个类别，还包括其他未标记的动物和车辆图像
    每个类含有800张测试样本
    96×96大小的RGB彩色图像
    '''
    ## Training settings
    if isVit:
        selfTransform = Aug_toTensor()
    else:
        selfTransform = Aug_RGBtoGray()
    trainset = datasets.STL10(root=path, split="train",download=True, transform=selfTransform)                
    testset = datasets.STL10(root=path, split="test", download=True, transform=selfTransform.test_transform)
    # 将训练集分为训练和测试集
    logging.info("==== load_STL10")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
#TODO: DTA
def load_SVHN(path="assets/datasets/SVHN", isVit=False):
    '''
    是RGB图像。针对RGB图像我们需要融合。
    街景门牌号 (SVHN) 数据集
    SVHN 是一个真实世界的图像数据集，用于开发机器学习和对象识别算法，
    对数据预处理和格式化的要求最低。它可以被视为与MNIST风格相似（例如，图像是经过裁剪的小数字），
    但包含一个数量级的更多标记数据（超过 600,000 个数字图像），
    并且来自一个更难、未解决的现实世界问题（识别自然场景图像中的数字和数字）。SVHN 是从谷歌街景图像中的门牌号获得的。
    '''
    if isVit:
        selfTransform = Aug_toTensor()
    else:
        selfTransform = Aug_RGBtoGray()
    trainset = datasets.SVHN(root=path, split='train', download=True, transform=selfTransform)                
    testset = datasets.SVHN(root=path, split='test', download=True, transform=selfTransform.test_transform)
    logging.info("==== load_SVHN")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)


#TODO :文本数据集
def load_reutersDataset(isVit=False,path='assets/datasets/REUTERS/reutersidf_total.h5'):
    x_train, x_test, y_train, y_test = GetReutersData(path)
    # x_train = normalize(x_train, axis=0, norm='max')
    # x_test = normalize(x_test, axis=0, norm='max')
    trainset = ReutersCustomDataset(X=x_train, y = y_train)
    testset  = ReutersCustomDataset(X=x_test, y = y_test, isTest=True)
    logging.info("==== load_Reuters")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
def load_reutersDataset20K(isVit=False,path='assets/datasets/REUTERS/reuters20K.h5'):
    x_train, x_test, y_train, y_test = GetReutersData(path)
    # x_train = normalize(x_train, axis=0, norm='max')
    # x_test = normalize(x_test, axis=0, norm='max')
    trainset = ReutersCustomDataset(X=x_train, y = y_train)
    testset  = ReutersCustomDataset(X=x_test, y = y_test, isTest=True)
    logging.info("==== load_Reuters")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)

def load_SOGOU10K(isVit=False,path='assets/datasets/SogouNews/sogouNews10K.h5'):
    '''
        因为数据集在拆分的时候做了，所以目录就行
    '''
    x_train, x_test, y_train, y_test = GetSogoData(path)
    # 将数据5标记变为标记1
    y_test[y_test==5.0]=0
    x_test[x_test==5.0]=0
    trainset = SoGoCustomDataset(X=x_train, y = y_train)
    testset  = SoGoCustomDataset(X=x_test, y = y_test, isTest=True)
    logging.info("==== load_SOGOU10K")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)

def load_AGNEWSDataset(isVit=False, path="./assets/datasets/AGNEWS/AGNEWs.h5",max_features=2000):
    x_train, x_test, y_train, y_test = getAgNews(path=path)
    trainset = AGNEWSCustomDataset(X=x_train, y = y_train)
    testset  = AGNEWSCustomDataset(X=x_test, y = y_test, isTest=True)
    logging.info("==== load_AGNEwS")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
def load_reutersDataset10K(isVit=False,path='assets/datasets/REUTERS/reuters10K.h5'):
    x_train, x_test, y_train, y_test = GetReutersData(path)
    # x_train = normalize(x_train, axis=0, norm='max')
    # x_test = normalize(x_test, axis=0, norm='max')
    # x_train = normalize(x_train)
    # y_train = normalize(y_train)
    #归一化处理
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0,1))
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test) 
    trainset = ReutersCustomDataset(X=x_train, y = y_train)
    testset  = ReutersCustomDataset(X=x_test, y = y_test, isTest=True)
    logging.info("==== load_Reuters")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
def load_reutersDataset20Knorm(isVit=False,path='assets/datasets/REUTERS/reuters20K.h5'):
    x_train, x_test, y_train, y_test = GetReutersData(path)
    # x_train = normalize(x_train, axis=0, norm='max')
    # x_test = normalize(x_test, axis=0, norm='max')
    # x_train = normalize(x_train)
    # y_train = normalize(y_train)
    #归一化处理
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test) 
    trainset = ReutersCustomDataset(X=x_train, y = y_train)
    testset  = ReutersCustomDataset(X=x_test, y = y_test, isTest=True)
    logging.info("==== load_Reuters")
    logging.info(f"Train Data: {len(trainset)}")
    logging.info(f"Test Data: {len(testset)}")
    return trainset, testset , len(testset)
# def load_AGNEWSDatasetnorm(isVit=False, path="./assets/datasets/AG_NEWS/",max_features=2000):
#     x_train, x_test, y_train, y_test = getAgNews(path=path)
#     # x_train = 
#     X=CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(x_train)
#     y= np.asarray(y_train)
#     X = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(X)
#     X = np.asarray(X.todense())*np.sqrt(X.shape[1])
#     p = np.random.permutation(X.shape[0])
#     X = X[p]
#     Y = y[p]
#     X2=CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(x_test)
#     y2= np.asarray(y_test)
#     X2 = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(X2)
#     X2 = np.asarray(X2.todense())*np.sqrt(X.shape[1])
#     p = np.random.permutation(X2.shape[0])
#     X2 = X2[p]
#     Y2 = y2[p]
#     x_train = X
#     y_train = Y
#     x_test = X2
#     y_test = Y2
#     y_train[y_train==4]=0
#     y_test[ y_test==4]=0
#     # x_train = normalize(x_train)
#     # y_train = normalize(y_train)
#     from sklearn.preprocessing import MinMaxScaler
#     scaler = MinMaxScaler(feature_range=(0,1))
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.fit_transform(x_test) 
#     trainset = AGNEWSCustomDataset(X=x_train, y = y_train)
#     testset  = AGNEWSCustomDataset(X=x_test, y = y_test, isTest=True)
#     logging.info("==== load_AGNEwS")
#     logging.info(f"Train Data: {len(trainset)}")
#     logging.info(f"Test Data: {len(testset)}")
#     return trainset, testset , len(testset)
