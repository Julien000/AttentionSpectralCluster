import torch.nn.functional as F
import torch
import math
from torch import nn
from scipy.sparse import csgraph
from utils.basic import top_k_eigvec_torch
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
    '''
     这个loss是有问题的：（1）这里的loss带有\ell_1，但却用到了\ell_{2,1}中的D，这是不对的。
                        （2）计算机loss的公式是错的，不能用梯度来计算
    write by liubo 2023-08-12
    '''
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
    '''
    这个loss跟L21LossV4()是一样的，只是L21LossV4是来自2022年8月份的老版本（存放在/backup/lwb/pytorch_sc/下）一样。
    这个loss是根据带有\ell_{2,1}正则化项的谱聚类来写实现的，通过大量的实验发现：self.beta对聚类效果影响很大，或者说D的对角线的值对聚类效果影响很大，具体而言，当对角线的值越小，聚类的效果越差，因此在计算对角线元素时，没有采用标准形式$1/2\|y^i\|_2$，而是直接采用$\|y^i\|_2$。这样不符合标准梯度的公式，但却可以得到较好的效果。
    '''
    def __init__(self,beta=1.0):
        super(L21LossV2, self).__init__()
        self.beta = beta
       
    def forward(self, Y, W,isVal=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        '''
        在self.beta=2的情况下，通过下面的方式计算D，可以达到最好的聚类ACC（在mnist上可以达到95%的acc）
        若采用标准的梯度（对角线元素按这个公式计算$1/2\|y^i\|_2$），效果很差。
        write by liubo 2023-08-12
        '''
        D=torch.diag((torch.norm(Y,p=2,dim=1)))    
        grads=  torch.mm(L,Y)+torch.mm(L.T,Y)+self.beta*torch.mm(D,Y)
        
        # 按这种方式计算loss是错的：loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0]
        # 下面这个才是基于L_{2,1}正则化的loss ，wirte by liubo 2023-08-11
        loss= (torch.trace(torch.mm(torch.mm(Y.T,L),Y))+self.beta*torch.sum(torch.norm(Y,p=2,dim=1)))/W.shape[0] # divde W
        
        return loss,grads
    
class L21LossV4(nn.Module):
    '''
       来自2022年8月份的老版本（存放在/backup/lwb/pytorch_sc/下）一样，它跟L21LossV2是一样的。当时为了作比较才增加的。
        write by liubo 2023-08-12
    '''
    def __init__(self,beta=1.0):
        super(L21LossV4, self).__init__()
        self.beta = beta
        
    def forward(self, Y, W, isVal=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        W = torch.from_numpy(W).to(device)      
        # D=torch.diag(1.0/2*(torch.norm(Y,p=2,dim=1)))
        # D=torch.diag(1.0/(torch.norm(Y,p=2,dim=1)))
        D=torch.diag((torch.norm(Y,p=2,dim=1))) # 效果最好。
        grads=torch.mm(L,Y)+torch.mm(L.T,Y)+torch.mm(D,Y)

        loss = torch.trace(torch.mm(torch.mm(Y.T,L),Y)) + self.beta*torch.sum(torch.norm(Y,p=2,dim=1)) 
        
        return loss, grads
    
class L21LossV3(nn.Module):
  # 论文中增加了收敛性之后，需要地算法修改。这个模块就是实现算法修改。
  # 主要修改内容：
  # （1）D不再按原来的公式计算。而是通过（L_t+D_{t-1}）的前k个特征向量构成矩阵，由这个矩阵的行来构建D_t的对角元素
  # （2）但在这个迭代过程中，t=1时，D_1的对角元素为1，在最后一个批次中，拉普拉斯矩阵L的大小不是batch_szie*batch_size，
  #     但D_{t-1}却是batch_szie*batch_size，这时需要对D_{t-1}进行处理。
  #  add by liubo 2023-08-10 
    '''
    为了增加论文的理论性，增加了算法迭代的过程，并且增加算法迭代收敛性的证明，但通过实验发现这样迭代的效果很差，ACC会随着迭代次数增加而讯速减少（在mnist数据集上，第一次迭代为60%，到第200次迭代时，acc为20%），因此，决定放弃这样的迭代。这个loss也放弃。
    add by liubo 2023-08-12
    '''  
    def __init__(self,beta=1.0,batch_size=2048,isDivW=False):
        super(L21LossV3, self).__init__()
        self.isDivW=isDivW
        self.beta = beta
        self.D = None
        
    def forward(self, Y, W, isVal=False):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 有这样的等价关系吗 YtLY =  YtL + LtY
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        if isVal:
            loss = torch.trace(torch.mm(torch.mm(Y.T,L),Y)) + self.beta*torch.sum(torch.norm(Y,p=2,dim=1)) 
            return loss, []
        Ysize = Y.shape[0]
        if self.D == None:
        # 针对第一个batch 不存在D_{t-1}
            matrix = torch.zeros(Ysize, Ysize)
            self.D = matrix.diagonal().fill_(1.0)
            self.D = self.D.to(device).double()
        D = []
        if Ysize != self.D.shape[0]:
            # 针对最后一个批次!= last batch D矩阵的维度的时候, 截取子矩阵。
            D = self.D[ :Ysize, :Ysize]  
        else:
            D  = self.D
        self.beta=1
        A=(L+self.beta* D)
        YLY = torch.mm(L,Y)+torch.mm(L.T,Y)
        # Y_hat=top_k_eigvec_torch(A, math.floor(Ysize/2))
        Y_hat=top_k_eigvec_torch(A, 10)
        D=torch.diag(1.0/(torch.norm(Y_hat,p=2,dim=1)+0.000001)) 
        DY=self.beta*torch.mm(D,Y)
        grads=YLY+DY
        # 这个loss是错的。
        loss =torch.abs(torch.sum(YLY+DY) )/ W.shape[0] 
        

        if Ysize != self.D.shape[0]:
            self.D[:Ysize, :Ysize]  = D [:Ysize, :Ysize] 
        else:
             self.D = D 
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



    



