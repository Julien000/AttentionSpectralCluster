from datetime import datetime
import logging
import os
import sys
import yaml
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import torch
import argparse
from sklearn.neighbors import kneighbors_graph,radius_neighbors_graph
from copy import copy
from utils.metrics import acc, get_rand_index_and_f_measure, purity
from torchvision import transforms
import numpy as np
HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
EXAMPLESDIR = os.path.dirname(HEREDIR)

def logger_init(log_file_name='monitor',
                 log_level=logging.DEBUG,
                 log_dir='./logInfo/',
                 only_file=False):
     # 指定路径
     if not os.path.exists(log_dir):
         os.makedirs(log_dir)
 
     log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
     formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
     if only_file:
         logging.basicConfig(filename=log_path,
                             level=log_level,
                             format=formatter,
                             datefmt='%Y-%d-%m %H:%M:%S')
     else:
         logging.basicConfig(level=log_level,
                             format=formatter,
                             datefmt='%Y-%d-%m %H:%M:%S',
                             handlers=[logging.FileHandler(log_path),
                                       logging.StreamHandler(sys.stdout)]
                             )
     print(f"loading logginfile in{log_file_name}")

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg
def cifar100SuperLabel(tar):
    super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
    ]
    tar_copy = copy.copy(tar)
    for i in range(20):
        for j in super_label[i]:
            tar[tar_copy == j] = i
    return tar

def to_graph_new(X,sigma,e,n_neighbors,similarity_matrix,knn_aprox,eps=1e-7):
    '''
    Compute similarity matrix.
      计算相似矩阵的模型
    return: similarity matrix
    '''
    if type(X) == torch.Tensor:
      X = X.detach().to("cpu").numpy()
    if len(X.shape)>2:
        X=X.reshape(X.shape[0],-1)
    # TODO :  代办的修改部分
    A = kneighbors_graph(X, n_neighbors, mode='distance',include_self=False, n_jobs=-1)
    if sigma == 'mean':
        sigma_2 = 2*np.power(A.sum(axis=1) / A.getnnz(axis=1).reshape(-1,1),2) + eps
        A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
        np.exp(-A.data,out=A.data)
    #近似解参数----后期考虑要不要可以优化速度。--目前未使用。
    if knn_aprox:
        A = A - sparse.identity(A.shape[0])
    # 新的相似度 处理方案。

    if similarity_matrix == 'k-hNNG':
        return (A + A.T)/2

def to_graph(X,sigma,e,n_neighbors,similarity_matrix,knn_aprox,eps=1e-7):
    '''
    Compute similarity matrix.
    return: similarity matrix
    '''
    if type(X) == torch.Tensor:
      X = X.detach().to("cpu").numpy()
      
    if similarity_matrix == 'e-NG':
      A = radius_neighbors_graph(X, e, mode='connectivity',include_self=False, n_jobs=-1)
      return A
    
    elif similarity_matrix == 'full':
        pass
    
    elif similarity_matrix == 'precomputed':
      return A

    else:
        
        if knn_aprox:
            A = PyNNDescentTransformer(n_neighbors=n_neighbors,metric="euclidean",n_jobs=-1).fit_transform(X)
        else:
            A = kneighbors_graph(X, n_neighbors, mode='distance',include_self=False, n_jobs=-1)
            
        if sigma == 'max':
            sigma_2 = 2*np.power(A.max(axis=1).toarray(),2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        elif sigma == 'mean':
            sigma_2 = 2*np.power(A.sum(axis=1) / A.getnnz(axis=1).reshape(-1,1),2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        else:
            sigma_2 = 2*np.power(sigma,2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        if knn_aprox:
            A = A - sparse.identity(A.shape[0])

        if similarity_matrix == 'k-hNNG':
            return (A + A.T)/2
            
        if similarity_matrix == 'k-NNG':
            return A.maximum(A.T)

        if similarity_matrix == 'k-mNNG':
            return A.minimum(A.T)


#TODO-代码--tokenCUt
def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    cls_token = feats[0,0:1,:].cpu().numpy() 

    feats = feats[0,1:,:]
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1,0)) 
    A = A.cpu().numpy()
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])

    # Using average point to compute bipartition 
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    
    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)



    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)

def eval_res(Y, tar, n_clusters,verbose=False,isRound=True,remainder_size=4):
    '''
    remainder_size: used to describe the length of metrics
    '''
    # 投影
    # Y = model(x).to('cpu').detach().numpy()
    # 计算原始spectralNet 的结果 出现了只有5个样本 大于7个类别的问题。
    kmeans = KMeans(n_clusters=n_clusters).fit(Y)
    NMI = normalized_mutual_info_score(tar,kmeans.labels_)
    ACC = acc(tar, kmeans.labels_)
    ARI=get_rand_index_and_f_measure(tar,kmeans.labels_)# 调整兰德系数 ，基于混淆矩阵
    Purity= purity(tar,kmeans.labels_) # 纯度指标

    # 通过将投影F ,直接求概率最大argmax()的结果
    argmaxPred = Y.argmax(1) # 预测结果的argmax
    argmaxNMI =  normalized_mutual_info_score(tar,argmaxPred)
    argmaxACC = acc(tar, argmaxPred)

    # 结果处理--这是中间处理的东西
    if isRound: # 需要round 的时候才round
        NMI = round(NMI, remainder_size)
        ACC = round(ACC, remainder_size)
        ARI = round(ARI, remainder_size)
        Purity = round(Purity, remainder_size)
        argmaxNMI = round(argmaxNMI, remainder_size)
        argmaxACC = round(argmaxACC, remainder_size)
    if verbose :
        print(f"******* ACC = {ACC} ,NMI = {NMI},ARI={ARI},Purity={Purity}  **********")
        print(f"******* argmax ACC = {argmaxACC} ,argmax NMI = {argmaxNMI} *******" )
    return ACC, NMI, ARI, Purity, argmaxACC, argmaxNMI

def eval_cifar(X,groundTrueY,n_clusters,model,verbose=False):
    # 读取模型的checkpoint 
    # model.load_state_dict(torch.load(args.PATH))
    expResultList = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #SpectralNet
    Y = model(X.to(device))

    # spectralNet _kmeans
    kmeans = KMeans(n_clusters=n_clusters).fit(Y.to('cpu').detach().numpy())
    NMI=round(normalized_mutual_info_score(groundTrueY,kmeans.labels_),4)
    ACC=round(acc(groundTrueY, kmeans.labels_),4)
    
    sp_label = Y.argmax(1).detach().to("cpu").numpy()
    argmaxNMI=round(normalized_mutual_info_score(groundTrueY, sp_label),4)
    argmaxACC=round(acc(groundTrueY, sp_label),4)

    expResultList.append([ 'spectralNet_kmeans', NMI, ACC])
    expResultList.append([ 'spectralNet_argmax(1)',argmaxNMI , argmaxACC])
    if verbose :
        print(f"******* NMI = {NMI},",f"ACC = {ACC} **********")
        print(f"******* argmax NMI = {argmaxNMI},",f"argmax ACC = {argmaxACC} *******" )
  
    return expResultList

def ChannelFussion(Y):
    '''
    该方法主要针对 3通道图像进行融合-并且/3
    '''
    return (Y[:,0,:]+Y[:,1,:]+Y[:,2,:])/3.0

def saveImage(img, path='./output/images'):
    toPIL = transforms.ToPILImage()
    pic = toPIL(img)
    pic.save(path)

def getDataInfo(idx="MNIST"):
    input_sizeMap={
        "MNIST": 28*28, #领先20%
        "QMNIST":28*28, #领先20%
        "Letters": 28*28, #领先3%
        "Digits": 28*28, #数据集简单，大家都差不多
        "FashionMNIST": 28*28, # 
        "USPS": 16*16, # 领先5%-10%
        #多通道图像部分
        "CIFAR10":32*32,  #领先3%
        "CIFAR100":32*32, 
        "STL10":96*96, #领先3%
        "SVHN": 32*32,
        "REUTERS": 2000,
        "REUTERS20K": 2000,
        "REUTERS10K": 2000,
        "SOGOU10K":2000,
        "AGNEWS": 2000,
        "REUTERS20Knorm": 2000,
        "AGNEWSnorm": 2000,
        "COIL": 128*128 # 领先5%-10%
    }
    n_clustersMap={ #类别字典

        "MNIST": 10,
        "QMNIST":10,
        "Letters": 26,
        "Digits": 10,
        "FashionMNIST": 10,
        "USPS": 10,
        "COIL":20,
        #多通道图像部分
        "CIFAR10":10,
        "CIFAR100":20,
        "STL10":10,
        "SVHN": 10,
        #文本数据集
        "REUTERS": 4,
        "REUTERS20K": 4,
        "REUTERS10K": 4,
        "AGNEWS": 4,
        "SOGOU10K": 5,
        "REUTERS20Knorm": 4,
        "AGNEWSnorm": 4
    }
    codeSizeMap={

        "MNIST": 10,
        "QMNIST":10,
        "Letters": 26,
        "Digits": 10,
        "FashionMNIST": 10,
        "USPS": 10,
        "COIL":20,
        #多通道图像部分
        "CIFAR10":10,
        "CIFAR100":20,
        "STL10":10,
        "SVHN": 10,
        #文本数据集
        "REUTERS": 10,
        "REUTERS20K": 10,
        "REUTERS10K": 10,
        "AGNEWS": 10,
        "SOGOU10K": 10,
        "REUTERS20Knorm": 10,
        "AGNEWSnorm": 10
    }
    # ModelMap={
    #     "Attention1":"AttentionNet1(input_size=args.code_size, output_size=args.n_clusters).to(device)", # 单层注意力网络
    #     "AttentionBest":"AttentionNet3(input_size=args.code_size, output_size=args.n_clusters).to(device)", # 目前最好的网络
    #     "AttentionNet":"AttentionNet(input_size=args.code_size, output_size=args.n_clusters).to(device)",# 纯注意力机制的网络
    #     "SpectralNetOld": "SpectralNet(input_size=args.code_size, output_size=args.n_clusters).to(device)", # 原始spectralNet 
    #     "SpectralNetL21": "SpectralNet(input_size=args.code_size, output_size=args.n_clusters).to(device)"  #基于L21的spectralNet -考虑不要那个2范数层是不是可以的
    # }
    return input_sizeMap[idx], n_clustersMap[idx], codeSizeMap[idx]

def get_logpath(suffix=""):
    """Create a logpath and return it.
    Args:
        suffix (str, optional): suffix to add to the output. Defaults to "".

    Returns:
        str: Path to the logfile (output of Cockpit).
    """
    save_dir = os.path.join(EXAMPLESDIR, "logfiles")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"cockpit_output{suffix}")
    return log_path

def save_hdf5(X, y, name):
    import h5py
    with h5py.File('./{}.h5'.format(name), 'w') as f:
        f['data'] = X
        f['labels'] = y

def seed_torch(seed=1):
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
 
