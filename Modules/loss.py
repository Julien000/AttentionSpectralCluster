import torch.nn.functional as F
import torch
from torch import nn
from scipy.sparse import csgraph
# L=csgraph.laplacian(W ,normed=False) 
# #W 代表权重矩阵，或者相似矩阵， normed ：用于选择归一化和非归一化 拉普拉斯矩阵
#TODO:1、 原始loss
class SpectralNetLoss(nn.Module):
    
    def __init__(self):
        super(SpectralNetLoss, self).__init__()      
    def forward(self, Y, W):
        Yd = torch.cdist(Y, Y, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')**2
        return torch.sum(W*Yd)/(W.shape[0]) 
class L1LossBeta(nn.Module):
    
    def __init__(self, beta=1.0,isDivW=False):
        super(L1LossBeta, self).__init__()
        self.isDivW=isDivW
        self.beta = beta
    def forward(self, Y, W,L ):
        #  对YY^T采用1范数
        return torch.trace(torch.mm(torch.mm(Y.T,L),Y))/(W.shape[0]) +self.beta*torch.sum(torch.norm(torch.mm(Y,Y.T),p=1)) /(W.shape[0])
#L1 loss
class L1LossV2(nn.Module):
    
    def __init__(self, beta=1.0,isDivW=False):
        super(L1LossV2, self).__init__()
        self.isDivW=isDivW
        self.beta = beta
    def forward(self, Y, W ):
        #  对YY^T采用1范数
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # 有这样的等价关系吗 YtLY =  YtL + LtY
        # L=csgraph.laplacian(W ,normed=False)
        # # L=torch.from_numpy(L).to(device).double()
        # L=torch.from_numpy(L).to(device).float()
        # YLY = torch.mm(L,Y)+torch.mm(L.T,Y)
        # D=torch.diag((torch.norm(Y,p=1,dim=1))) # 计算L1范数
        # DY=self.beta*torch.mm(D,Y)
        # loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0]
        # grads=YLY+DY
        # return loss,grads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        # L=torch.from_numpy(L).to(device).float()
        YLY = torch.mm(L,Y)+torch.mm(L.T,Y)
        D=torch.diag((torch.norm(Y,p=1,dim=1))) # 计算L1范数
        DY=self.beta*torch.mm(D,Y)
        loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0]
        grads=YLY+DY
        return loss,grads
    
class L0Loss(nn.Module):
    
    def __init__(self, beta=1.0,isDivW=False):
        super(L0Loss, self).__init__()
        self.isDivW=isDivW
        self.beta = beta
    def forward(self, Y, W ):
        #  对YY^T采用1范数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        # L=torch.from_numpy(L).to(device).float()
        YLY = torch.mm(L,Y)+torch.mm(L.T,Y)
        D=torch.diag((torch.norm(Y,p=0, dim=1))) # 计算L1范数
        DY=self.beta*torch.mm(D,Y)
        loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0]
        grads=YLY+DY
        return loss,grads
class L2Loss(nn.Module):
    def __init__(self, beta=1.0,isDivW=False):
        super(L2Loss, self).__init__()
        self.isDivW=isDivW
        self.beta = beta
    def forward(self, Y, W ):
        #  对YY^T采用1范数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        # L=torch.from_numpy(L).to(device).float()
        YLY = torch.mm(L,Y)+torch.mm(L.T,Y)
        D=torch.diag((torch.norm(Y,p=0, dim=1))) # 计算L1范数
        DY=self.beta*torch.mm(D,Y)
        loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0]
        grads=YLY+DY
        return loss,grads
class NoNormLoss(nn.Module):
    
    def __init__(self, beta=1.0,isDivW=False):
        super(NoNormLoss, self).__init__()
        self.isDivW=isDivW
        self.beta = beta
    def forward(self, Y, W ):
        #  对YY^T采用1范数

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        # L=torch.from_numpy(L).to(device).float()
        YLY = torch.mm(L,Y)+torch.mm(L.T,Y)
        D=torch.diag(torch.diag(Y)) # 计算L1范数
        # DY=torch.mm(D,Y)
        DY=torch.mm(Y,D)
        loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0]
        grads=YLY+DY
        return loss,grads
#TODO: L21 loss
class L21LossV2(nn.Module):
    
    def __init__(self,beta=1.0,isDivW=False):
        super(L21LossV2, self).__init__()
        self.isDivW=isDivW
        self.beta = beta
        
    def forward(self, Y, W):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        # L=torch.from_numpy(L).to(device).float()
        YLY = torch.mm(L,Y)+torch.mm(L.T,Y)
        D=torch.diag((torch.norm(Y,p=2,dim=1))) 
        DY=self.beta*torch.mm(D,Y)
        loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0]
        grads=YLY+DY
        return loss,grads
    
class L21Loss_D(nn.Module):
    
    def __init__(self,beta=1.0,batch_size=2048,isDivW=False):
        super(L21Loss_D, self).__init__()
        self.isDivW=isDivW
        self.beta = batch_size
        self.D = None
        
    def forward(self, Y, W):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        Ysize = Y.shape[0]
        if self.D == None:
        # 针对第一个batch 不存在D t-1
            matrix = torch.zeros(Ysize, Ysize)
            self.D = matrix.diagonal().fill_(1.0)
            self.D = self.D.to(device).double()

        if Ysize != self.D.shape[0]:
            # 针对最后一个批次!= last batch D矩阵的维度的时候, 截取子矩阵。
            sub_D = self.D[:Ysize, :Ysize]
            A = (L+sub_D)
            YLY = torch.mm(A,Y)+torch.mm(A.T,Y)
            D=torch.diag((torch.norm(Y,p=2,dim=1))) 
            self.D[:Ysize, :Ysize] = D
        else:      
            A = (L+self.D)
            YLY = torch.mm(A,Y)+torch.mm(A.T,Y)
            D=torch.diag((torch.norm(Y,p=2,dim=1))) 
            self.D = D

        DY=self.beta*torch.mm(D,Y)
        loss =torch.abs(torch.sum(YLY+DY) )
        grads=YLY+DY
        return loss,grads

class L21LossV3(nn.Module):
    
    def __init__(self,beta=1.0,isDivW=False):
        super(L21LossV2, self).__init__()
        self.isDivW=isDivW
        self.beta = beta
        # self.lastD = torch.
        
    def forward(self, Y, W):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        # L=torch.from_numpy(L).to(device).float()
        YLY = torch.mm(L,Y)+torch.mm(L.T,Y)
        # D=torch.diag((torch.norm(Y,p=2,dim=1)))  # V1
        D=torch.diag(1.0/(torch.norm(Y,p=2,dim=1))) 
        DY=self.beta*torch.mm(D,Y)
        # loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0]
        loss =torch.abs(torch.sum(YLY+DY) )
        grads=YLY+DY
        return loss,grads

# 这个网络是给孪生网络使用的参数。
class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, z1, z2, target):
        distances = (z2 - z1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances + (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()

class YtXYloss(nn.Module):
    def __init__(self):
        super(YtXYloss, self).__init__()
        self.loss2 = nn.CrossEntropyLoss()

    def forward(self, X, Y):

        YXY  = Y @ X @Y.t
        yxyloss = torch.norm(X-YXY,p=2)
        
        loss2 = self.loss2(X,YXY )

        return torch.sum(yxyloss)/X.shape[0] + 0.5*loss2

class BlockDiagLoss(nn.Module):
    def __init__(self):
        super(BlockDiagLoss,self).__init__()
    def forward(self,Y,X):
        '''
            X = [b, feat]
            Y = [b, low_feat]
           YtY = [feat*feat]
           X * YtY = [b,feat]
        当 B 的对角线==0 , B >= 0 ,B=B.t B要对称三个条件 ,比较苛刻.
        测试时间: 2022-10-19
        测试结果: 目前能得到的最好效果,为ACC =  0.35 是当前的结果.然后精度不会得到提升.所以尝试的关系是有问题的
        针对性的采用21范数,ACC ~ 0.4 效果也是不好的.
        
        测试效果: 差

        '''
        # 目前这里两个不同的方案, 首先
        # T1,  X 代表样本
        # T2 , X 代表 W 相似矩阵

        B = torch.mm( Y.T , Y)
        XB =torch.mm( X , B)
        norm= torch.norm(X-XB, p=2,dim=1)
        loss = torch.sum(1.0/2*norm)/X.shape[0]
        grads =  1.0/2*norm
        return loss, grads
class BlockDiagLossW(nn.Module):
    def __init__(self):
        super(BlockDiagLossW,self).__init__()
    def forward(self,Y,X):
        '''
            X = [b, feat]
            Y = [b, low_feat]
           YtY = [feat*feat]
           X * YtY = [b,feat]
        当 B 的对角线==0 , B >= 0 ,B=B.t B要对称三个条件 ,比较苛刻.
        实验时间:2022- 10 -19
        实验描述: 根据W 权重矩阵,计算loss ,效果并不好,
        呈现出来的特点是,这个损失比较小,但是可以不用除去样本试试
        '''
        # 目前这里两个不同的方案, 首先
        # T1,  X 代表样本
        # T2 , X 代表 W 相似矩阵
        B = torch.mm( Y , Y.T)
        XB =torch.mm( X , B)
        loss= 1.0/2*torch.norm(X-XB, p=2)
        grads =  loss
        return loss, grads



    



