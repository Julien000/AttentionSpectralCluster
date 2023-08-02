from utils.basic import ChannelFussion, cifar100SuperLabel
import torch
from utils.metrics import acc, get_rand_index_and_f_measure, purity
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np
def evalBasicRes(Y, tar, n_clusters):
    '''
    remainder_size: used to describe the length of metrics
    '''
    X=Y.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    pred = kmeans.labels_
    NMI = normalized_mutual_info_score(tar,pred)
    ACC = acc(tar, pred)
    ARI=get_rand_index_and_f_measure(tar,pred)# 调整兰德系数 ，基于混淆矩阵
    Purity= purity(tar,pred) # 纯度指标
    return ACC, NMI, ARI, Purity
def evalBasicRes_text(Y, tar, n_clusters):
    '''
    remainder_size: used to describe the length of metrics
    '''
    X=Y
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    pred = kmeans.labels_
    NMI = normalized_mutual_info_score(tar,pred)
    ACC = acc(tar, pred)
    ARI=get_rand_index_and_f_measure(tar,pred)# 调整兰德系数 ，基于混淆矩阵
    Purity= purity(tar,pred) # 纯度指标
    return ACC, NMI, ARI, Purity

def evaluationForImage(test_dataloader, model, model_ae,  args, isold=False ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ACC_T, NMI_T, ARI_T, P_T= 0,0,0,0
    for x,tar in test_dataloader:
        bs, c, w, h = x.shape
        x= x.to(device)
        #embedding
        if args.embedding:
            if(c>1):
                x = x.reshape( bs, c, -1).to(device)
            else:
                x =x.reshape( bs, -1).to(device)
            if isold==False: # 对注意力增加通道
                x =x.reshape( bs,1, -1)
            x =  model_ae.encoder(x.float()).double()
            # x =  model_ae.encoder(x)
        Y = model(x)
        if(c>1 and args.isVit==False):#通道合并。
            Y = ChannelFussion(Y) 
        if args.data == "cifar100":  # super-classcifar 含有超类
            tar = cifar100SuperLabel(tar)
        if isinstance(Y, tuple):
            ACC, NMI, ARI, P= evalBasicRes( Y[0], tar.numpy(), args.n_clusters)
        else:
            ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
        ACC_T+=ACC
        NMI_T += NMI
        ARI_T += ARI
        P_T += P
    return round(ACC_T/len(test_dataloader),4), round(NMI_T/len(test_dataloader),4), round(ARI_T/len(test_dataloader),4), round(P_T/len(test_dataloader),4)
    
def evaluationForText(test_dataloader, model, model_ae,  args, isold=True ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ACC_T, NMI_T, ARI_T, P_T=0,0,0,0
    for x,tar in test_dataloader:
        bs, c= x.shape
        x= x.to(device)
        #embedding
        if args.embedding:
            if isold==False: # 对注意力增加通道
                x =x.reshape( bs,1, -1)
            x =  model_ae.encoder(x.float()).double()
        Y = model(x)
        Y=Y.reshape(bs,-1)
        ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
        ACC_T+=ACC
        NMI_T += NMI
        ARI_T += ARI
        P_T += P
    return round(ACC_T/len(test_dataloader),4), round(NMI_T/len(test_dataloader),4), round(ARI_T/len(test_dataloader),4), round(P_T/len(test_dataloader),4)
    
def evaluationForText_Infer(test_dataloader, model, model_ae, datacont, args, isold=False ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X, y = [], []
    for x,tar in test_dataloader:
        bs, c = x.shape
        x= x.to(device)
        #embedding
        if args.embedding:
            x =x.reshape( bs,1, -1).to(device)
            x =  model_ae.encoder(x.float()).double()
        Y = model(x)
        Y=Y.reshape(bs,-1)
        X.append(Y.detach().to('cpu').numpy())
        y.append(tar.numpy())    
        ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
        datacont.setdata(data=[ACC,NMI,ARI,P])
    return np.array(X).reshape(bs,-1), np.array(y).reshape(-1)  

def evaluationForImage_Infer(test_dataloader, model, model_ae, datacont, args, isold=False ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X, y = [], []
    for x,tar in test_dataloader:
        bs, c, w, h = x.shape
        x= x.to(device)
        #embedding
        if args.embedding:
            if(c>1):
                x = x.reshape( bs, c, -1).to(device)
            else:
                x =x.reshape( bs, -1).to(device)
            if isold==False: # 对注意力增加通道
                x =x.reshape( bs,1, -1)
            x =  model_ae.encoder(x.float()).double()
        Y = model(x)
        if(c>1 and args.isVit==False):#通道合并。
            Y = ChannelFussion(Y) 
        if args.data == "cifar100":  # super-classcifar 含有超类
            tar = cifar100SuperLabel(tar)
        Y=Y.reshape(bs,-1)
        X.append(Y.detach().to('cpu').numpy())
        y.append(tar.numpy())    
        ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
        datacont.setdata(data=[ACC,NMI,ARI,P])
    return np.array(X).reshape(bs,-1), np.array(y).reshape(-1)  

