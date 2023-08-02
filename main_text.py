import argparse
import logging
import os
from Modules.train.trainL21Loss_text import *
from Modules.train.trainOldLoss_text import *
from utils.basic import getDataInfo, logger_init, yaml_config_hook
from torch.utils.data import DataLoader
from utils import exportExcel
from Modules.train.train_AE import *
from Modules.train.trainL21Loss import *
from Modules.train.trainOldLoss import *
from utils.loadData import * # 加载数据集模块

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

def main(args):    
    datasets = args.datasets
    trains = args.trains
    dataMap={
        "REUTERS": load_reutersDataset,
        "REUTERS20K": load_reutersDataset20K, #20K 是REUTERS的一个 tf-idf 特征编码
        "REUTERS10K": load_reutersDataset10K, #10K 是REUTERS的一个子集 #包含归一化操作
        "AGNEWS": load_AGNEWSDataset, # it-idf特征编码,
        "SOGOU10K": load_SOGOU10K,
        # "REUTERS20Knorm": load_reutersDataset20Knorm, #20K 是REUTERS的一个子集 #包含归一化操作
        # "AGNEWSnorm": load_AGNEWSDatasetnorm # 包含norm包含归一化操作
    }
    trainTextMap={
        "SpectralNet_oldLoss": train_SpectralNet_oldLoss_text, # 原始spectralNnet
        "AttentionN_L21": train_SpectralNet_attNet_text, # 纯注意力网络
        # "SpectralNet_L21": train_SpectralNet_L21_text, # 原始spectralNet 和L21范数
        # "Attention1_L21": train_SpectralNet_att1_text, # 单层注意力层
        # "Attention_L21": train_SpectralNet_att3_text, #多层注意力和单层全连接
    }

    datalist = []
        
    for _, data_idx in enumerate(datasets):# 加载所有的数据集
        # 加载数据集
        trainsets, testsets, test_batch_size = dataMap[data_idx](isVit=args.isVit)
        # args.test_batch_size = test_batch_size
        train_dataloader = DataLoader(trainsets, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(testsets, batch_size=args.test_batch_size, shuffle=False)
        # 更新模型参数
        input_size, n_clusters,code_size = getDataInfo(data_idx)
        args.input_size = input_size
        args.code_size = n_clusters # 编码到潜孔间表示也是非常复杂的一环想要怎么做才行。
        args.n_clusters = n_clusters
        args.data = data_idx
        logging.info(f"dataset=={data_idx},n_clusters={n_clusters}, input_size={input_size}")
        #加载自编码器模块
        if args.embedding:
            model_ae = load_AE(train_dataloader=train_dataloader, isText=True, test_dataloader=test_dataloader,dataName=args.aedir+data_idx,args=args)
        for _,train_model_idx in enumerate(trains): # 加载所有训练模型。
            #训练模型，评估模型部分。
            for _, iter_beta in enumerate(args.betaList):
                args.beta = iter_beta
                logging.info(f"=================Beta={iter_beta}==codesize{args.code_size}")
                args.model_type = train_model_idx+data_idx+"_beta"+str(iter_beta)
                result = trainTextMap[train_model_idx](train_dataloader,test_dataloader, model_ae, args=args)
                result.append(iter_beta)
                datalist.append(result)
    # 记录结果
    # 14个列 导出结果。
    cols=('model','epoch', "datasets",'ACC', 'NMI','ARI', 'purity', 'AVG_ACC', "AVG_NMI","AVG_ARI", "AVG_purity","n_clusters",'learningRate','batch_size',"beta")
    exportExcel.export_excel(sheet_name=args.outExeName+args.log_fileName, col=cols, datalist=datalist)
    logging.info("====== end executive ！！！ ==========")

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    #固定的配置部分
    basic_config = yaml_config_hook("assets/config/config_basic.yaml")
    for k, v in basic_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument("--config", default="config_active_text.yaml", type=str, help="loading the model parameters type=str ")
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

    #固定网络模型的参数。--为了实验能够更好的复现。
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    main(args)
    