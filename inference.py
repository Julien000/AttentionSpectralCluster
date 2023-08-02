import argparse
import logging
import os
from Modules.evaluation import evaluationForImage, evaluationForImage_Infer
from Modules.train.train_AE import load_AE
from assets.datasets.coil20 import load_coil20
from utils import exportExcel
from utils.basic import logger_init, yaml_config_hook
from utils.data_contrainer import dataContainer
from utils.loadData import *
# from tranditionalClustering import test_cluster
from torch.utils.data import DataLoader
from utils.basic import getDataInfo
from utils.visualization.tsne_visulizer import VisTSNE
from utils.visualization.vis_grid import visGrid
from utils.visualization.viz_model import *
import matplotlib.pyplot as plt
import random
import torch.utils.data as data

def main(args):
    '''
    # 1加载数据集
    # 2、参数设置
    # 3、 加载模型
    # 4、 将数据送入模型-->得到结果
    # 5、 将结果与target 用于评估
    
, "QMNIST", "Letters" , "Digits","FashionMNIST", "USPS"
    '''
    datas =["MNIST"]
    dataMap={
        "MNIST": load_mnist,
        "QMNIST":load_qmnist,
        "Letters": load_Letters,
        "Digits": load_digits,
        "FashionMNIST": load_fashionMinst,
        "USPS": load_USPS,
        "COIL":load_coil20,
        #多通道图像部分
        "CIFAR10":load_cifar10,
        "CIFAR100":load_cifar100,
        "STL10":load_stl10,
        "SVHN": load_SVHN
    }
    PreModelMap=[
        # "AttentionN_L21MNIST_beta4_MNIST.pkl",
        # "AttentionN_L21_beta_0_MNIST.pkl",
        # "AttentionN_L21_beta_0.5_MNIST.pkl",
        # "AttentionN_L21_beta_1_MNIST.pkl",
        "AttentionN_L21_beta_2_MNIST.pkl",
        # "AttentionN_L21_beta_3_MNIST.pkl",
        # "AttentionN_L21_beta_4_MNIST.pkl",
        # "AttentionN_L21_beta_5_MNIST.pkl",
    #    "Spectral_oldloss_QMNIST.pkl",
    #    "SpectralNet_oldLoss_MNIST.pkl",
    #    "SpectralAttention_best_oldLoss_QMNIST.pkl"
    ]
    contra_model={
        "DEKM":"", #
        "dec":"",
        # "spectralNet":"",# 在实验过程中，单独复现过
        "DCN":"",#Towards K-means-friendly spaces: Simultaneous deep learning and clustering https://github.com/boyangumn/DCN-New
        "DSCDAN":"",#"Deep-Spectral-Clustering-using-Dual-Autoencoder-Network" # https://github.com/xdxuyang/Deep-Spectral-Clustering-using-Dual-Autoencoder-Network
        "DCULVF":"",#Deep Clustering for Unsupervised Learning of Visual Features https://github.com/facebookresearch/deepcluster,
        "DeepSpectralClusteringLearning":""
    }

    datalist = []
    for _, data_idx in enumerate(args.datasets):# 加载所有的数据集
        trainsets, testsets, test_batch_size = dataMap[data_idx](isVit=False)

        # test_batch_size =  args.test_batch_size
        indices = list(range(len(testsets)))
        subset_indices = random.sample(indices, test_batch_size)  # 随机选择10000个数据样本的索引
        subset_dataset = data.Subset(testsets, subset_indices)

        test_dataloader = data.DataLoader(subset_dataset, batch_size=test_batch_size, shuffle=False)
        # test_dataloader = DataLoader(testsets, batch_size=test_batch_size, shuffle=False)
        input_size, n_clusters, code_size = getDataInfo(data_idx)
        args.input_size = input_size
        args.code_size = n_clusters # 编码到潜孔间表示也是非常复杂的一环想要怎么做才行。
        args.n_clusters = n_clusters
        args.data = data_idx
        # 默认采用embedding
        if True:
            model_ae = load_AE(test_dataloader,dataName=args.aedir+data_idx,args=args)
        for _2,test_model_idx in enumerate(PreModelMap):
            datacont = dataContainer(len=4*3)
            model = torch.load(args.inferDir+test_model_idx) #直接加载模型的结果。
            model.eval()
            model_ae.eval()
            testSize = len(test_dataloader)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vis_name = data_idx+test_model_idx.split('.',1)[0]+".svg"
            if True:
                AE_X, AE_y = [], []
                X,Y = [], []
                for x,tar in test_dataloader:
                    bs, c, w, h = x.shape
                    x= x.to(device)
                    # x =x.reshape( bs,1, -1)
                    X.append(x.detach().to('cpu').numpy())
                    Y.append(tar.numpy())
                # VisTSNE(X[0], Y[0], test_size=test_batch_size, outdir=args.vizdir, outName="TSNE_row_mnist.svg")
                for x,tar in test_dataloader:
                    bs, c, w, h = x.shape
                    x= x.to(device)
                    x =x.reshape( bs,1, -1)
                    x =  model_ae.encoder(x.float()).double()
                    AE_X.append(x.detach().to('cpu').numpy())
                    AE_y.append(tar.numpy())
                VisTSNE(AE_X[0], AE_y[0], test_size=test_batch_size, outdir=args.vizdir, outName="TSNE_AE_"+vis_name)
            X,y =evaluationForImage_Infer(test_dataloader=test_dataloader, model=model, model_ae=model_ae, datacont=datacont, args=args)
            # # 1、可视化 TSNET图片
            VisTSNE(X,y, test_size=test_batch_size, outdir=args.vizdir, outName="TSNE"+vis_name)
            # 2、 可视化 GRid 图像
            # visGrid(X,n_clusters=args.n_clusters, fileName=args.vizdir+"Grid"+vis_name)
            # 3、 数据网络架构图.
            # data ,target  = next(iter(test_dataloader))
            # bs=10
            # X1 = data[0:bs].reshape(bs,1,-1)
            # X1 = model_ae.encoder(X1.cuda().float()).double()
            # Y1 = model(X1)
            # try:
            #     vizModel(X1,Y1,model,dir=args.vizdir+"model_arch_"+vis_name.split('.',1)[0])
            # except:
            #     print("pass error")
            # 4、 数据架构信息.            
            logging.info(f"======model{vis_name.split('.',1)[0]}============")
            # inputSize=X1.shape
            # modelInfo=vizModelInfo(model=model, inputSize=inputSize)
            # logging.info(modelInfo)


            ACC, NMI, ARI, P = datacont.GetOneEpochResult(size=testSize)
            logging.info(f"method = {test_model_idx}, datasets={data_idx},ACC={ACC},NMI={NMI},ARI={ARI},Purity={P} ")
            row=[test_model_idx, _, ACC, NMI, ARI, P]
            datalist.append(row)  
        #对比算法部分          
        # for _2,test_model_idx in enumerate(contra_model):
        #     datacont = dataContainer(len=4*3)
        #     model = torch.load(test_model_idx) #直接加载模型的结果。
        #     model.eval()
        #     evaluationForImage(test_dataloader=test_dataloader, model=model, model_ae=model_ae, datacont=datacont, args=args)
        #     ACC, NMI, ARI, P = datacont.getdata
        #     logging.info(f"method = {test_model_idx}, datasets={data_idx},ACC={ACC},NMI={NMI},ARI={ARI},Purity={P} ")
        #     row=[_2, _, ACC, NMI, ARI, P]
        #     datalist.append(row)
        #传统聚类算法部分的结果输出。
        # X, y = next(iter(test_dataloader))
        # X=X.reshape(test_batch_size,-1).detach().numpy()
        # y = y.numpy()
        # VisTSNE(X, y, outdir=args.logdir, test_size=test_batch_size, outName=data_idx+"_rowdata.jpg")
        # # [['Kmeans', 'MNIST', 0.5013850987802866, 0.5467, 0.38099932644261486, 0.5922], ['AgglomerativeClustering', 'MNIST', 0.7114404571346596, 0.6948, 0.6063391081376247, 0.733], ['DBSCAN', 'MNIST', 0.2355473406601777, 0.2055, 0.04218661870718442, 0.2209]]
        # rows = test_cluster(X, y , n_clusters , data_idx)
        # for i in rows:
        #     datalist.append(i)

    # 导出评估结果
    cols=('method', "datasets",'ACC', 'NMI','ARI', 'purity')
    exportExcel.export_excel(sheet_name=args.outExeName, col=cols, datalist=datalist) 

    logging.info("============= evaluation end!!!!!==============")

        

    # 传统聚类方法
    # tran_res_list = test_cluster(X,y, k=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="batch_config_Inference.yaml", type=str, help="loading the model parameters type=str ")
  
    config = yaml_config_hook("assets/config/"+parser.parse_args().config)
    for k, v in config.items():
      parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
  
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.file = args.file if args.file!='' else None 
    args.lr =eval(args.lr)
    logFileName = args.config
    logger_init(log_file_name=logFileName, log_dir=args.logdir)
    logging.info("========staring exective the programing==========")
    logging.info(f"=====parameters:{args}")
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    main(args)
