import argparse
import logging
import os
from tabnanny import verbose
from Modules.train.train_SiameNet import siamesedataloader, train_SiameseNet
from assets.datasets.coil20 import load_coil20
from utils.basic import getDataInfo, logger_init, seed_torch, yaml_config_hook
from torch.utils.data import DataLoader
from utils import exportExcel
from Modules.train.train_AE import *
from Modules.train.trainL21Loss import *
from Modules.train.trainOldLoss import *
from utils.loadData import * # 加载数据集模块

import sys

from utils.siameseData import CustomSiamesesDataset, SiameseDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

def main(args):
    
    datasets = args.datasets
    trains = args.trains
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
    trainMap={
        "SpectralNet_oldLoss": train_SpectralNet_oldLoss, # 原始spectralNnet
        "Attention1_L21": train_SpectralNet_att1, # 单层注意力层
        "Attention_L21": train_SpectralNet_att3, # 最优的注意力层
        "AttentionN_L21": train_SpectralNet_attNet, # 纯注意力网络
        "SpectralNet_L21": train_SpectralNet_L21, # 原始spectralNet 和L21范数
        "Test": train_SpectralNet_Test # 原始spectralNet 和L21范数
    }

    datalist = []
        
    for _, data_idx in enumerate(datasets):# 加载所有的数据集
        # logging.info
        # 加载数据集
        trainsets, testsets, test_batch_size = dataMap[data_idx](isVit=args.isVit)
        args.test_batch_size = test_batch_size
        train_dataloader = DataLoader(trainsets, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(testsets, batch_size=args.batch_size, shuffle=False)
        # 更新模型参数
        input_size, n_clusters,code_size  = getDataInfo(data_idx)
        args.input_size = input_size
        args.code_size = n_clusters # 编码到潜孔间表示也是非常复杂的一环想要怎么做才行。
        args.n_clusters = n_clusters
        args.data = data_idx
        logging.info(f"dataset=={data_idx},n_clusters={n_clusters}, input_size={input_size}")
        #加载自编码器模块
        if args.embedding:
            model_ae = load_AE(train_dataloader=train_dataloader,test_dataloader =test_dataloader,dataName=args.aedir+data_idx,args=args)
        if args.siaming:
            #孪生网络
            '''
            1. 将当前的数据全都放在一起，
            2. 生成孪生网络的结果。
            '''
            # siamfiledir = args.siamdir+data_idx+"_"+"inSize"+".pkl"
            input_size_siam = args.code_size
            output_size_siam  = args.n_clusters
            siamfiledir = f"{args.siamdir+data_idx}_inputsize{input_size_siam}.pkl"
            if os.path.exists(siamfiledir):
                model_siam = torch.load(siamfiledir)
            else:    
                n_neigbors_siam = 2
                user_approx = True
                
                # 目前针对的是code——size 版本的孪生网络
                X1 = np.array([trainsets[i][0][0][0].numpy() for i in range(len(trainsets))])
                X2 = np.array([testsets[i][0][0].numpy() for i in range(len(testsets))])
                X = np.concatenate((X1, X2), axis=0)
                X = X.reshape(-1, input_size)
                X = model_ae.encoder(torch.from_numpy(X).cuda().float()).double().detach().cpu().numpy()
                verbose = True
                train_siamDataloder, test_siamDataloader = siamesedataloader(X, n_neigbors_siam,batch_size=args.batch_size, use_approx=user_approx)
                model_siam = train_SiameseNet( input_size_siam , output_size_siam,train_siamDataloder, test_siamDataloader,file=siamfiledir,epochs=args.AE_epochs, verbose=verbose )
        else:
            model_siam=None
        for _,train_model_idx in enumerate(trains): # 加载所有训练模型。
            #训练模型，评估模型部分。
            #TODO:对模型的beta值进行调整。
            for _, iter_beta in enumerate(args.betaList):
                args.beta = iter_beta
                logging.info(f"=================Beta={iter_beta}==codesize{args.code_size}")
                args.model_type = train_model_idx+data_idx+"_beta"+str(iter_beta)
                result = trainMap[train_model_idx](train_dataloader,test_dataloader, model_ae,model_siam=model_siam, args=args)
                result.append(iter_beta)
                datalist.append(result)
                if train_model_idx == "SpectralNet_oldLoss": 
                    # 对于 不需要超参数 的模型只在超参数 --- 只用执行一次循环就可以结束了 
                    break
    # 记录结果
    # 14个列 导出结果。
    cols=('model','epoch', "datasets",'ACC', 'NMI','ARI', 'purity', 'AVG_ACC', "AVG_NMI","AVG_ARI", "AVG_purity","n_clusters",'learningRate','batch_size',"beta")
    exportExcel.export_excel(sheet_name=args.outExeName+args.log_fileName, col=cols, datalist=datalist)
    #TODO: 评估传统聚类方法，评估深度聚类模型方法。
    logging.info("====== end executive ！！！ ==========")

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    #固定的配置部分
    basic_config = yaml_config_hook("assets/config/config_basic.yaml")
    for k, v in basic_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument("--config", default="config_active_image.yaml", type=str, help="loading the model parameters type=str ")
    config = yaml_config_hook(parser.parse_args().cofigdir+parser.parse_args().config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
  
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.file = args.file if args.file!='' else None 
    args.lr =eval(args.lr)
    logFilename =args.config+"_"+args.log_fileName
    logger_init(log_file_name=logFilename, log_dir=args.logdir)
    logging.info("========staring exective the programing==========")
    logging.info(f"=====parameters:{args}")

    seed_torch()#固定网络模型的参数。--为了实验能够更好的复现。
    main(args)
    