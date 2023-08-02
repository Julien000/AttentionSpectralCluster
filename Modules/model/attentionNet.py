
import torch
from torch import nn
from torch.nn.functional import normalize
from Modules.model.selfAttention import MultiHeadAttention

class AttentionNet1(nn.Module):
    '''
        包含单个注意力层,
        norm_layer=None,
        act_layer=None,
    '''
    def __init__(self, **kwargs):
        super(AttentionNet1, self).__init__()
        self.s = nn.Sequential(
            MultiHeadAttention(embed_dim=kwargs["input_size"], num_hiddens=16, num_heads=8,  isMid=False),
            nn.Linear(16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,kwargs["output_size"]),
            nn.ReLU())
        for name,param in self.s.named_parameters():
            param.data = param.data.double()

    def make_ortho_weights(self,x):
        eps= 1e-7
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_sym = torch.mm(x.t(), x) 
        x_sym += torch.eye(x_sym.shape[1],device=device)*eps
        L = torch.linalg.cholesky(x_sym)
        ortho_weights = ((x.shape[0])**(1/2) )* (torch.inverse(L)).t()
        return ortho_weights
    
    def ortho_update(self,x):
        ortho_weights = self.make_ortho_weights(x)
        self.W_ortho = ortho_weights
        
    def forward(self, x, ortho_step=False):
        # x_net = normalize(self.s(x), dim=1)
        x_net = self.s(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho               
        return y
    
class AttentionNet3(nn.Module):
    '''
        add by :liub
        for mnist
        包含单个3个注意力层网络,加上一个线性层，根据老师的结构复现的网络。
        ---目前这个结构是效果最优的
    '''
    def __init__(self, **kwargs):
        super(AttentionNet3, self).__init__()
        self.s = nn.Sequential(
            MultiHeadAttention(embed_dim=kwargs["input_size"], num_hiddens=512, num_heads=8, dropout=0.1),
            MultiHeadAttention(embed_dim=512, num_hiddens=256, num_heads=4, dropout=0.1),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=256, num_hiddens=16, num_heads=4,  isMid=False),
            nn.ReLU(),
            nn.Linear(16,kwargs["output_size"]),
            nn.ReLU()
        )
        for name,param in self.s.named_parameters():
            param.data = param.data.double()

    def make_ortho_weights(self,x):
        eps= 1e-7
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_sym = torch.mm(x.t(), x) 
        x_sym += torch.eye(x_sym.shape[1],device=device)*eps
        L = torch.linalg.cholesky(x_sym)
        ortho_weights = ((x.shape[0])**(1/2) )* (torch.inverse(L)).t()
        return ortho_weights
    
    def ortho_update(self,x):
        ortho_weights = self.make_ortho_weights(x)
        self.W_ortho = ortho_weights
        
    def forward(self, x, ortho_step=False):
        # x_net = normalize(self.s(x), dim=1)
        x_net = self.s(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho               
        return y
  
class AttentionNet(nn.Module):
    def __init__(self, **kwargs):
        super(AttentionNet, self).__init__()
        self.s = nn.Sequential(
            MultiHeadAttention(embed_dim=kwargs["input_size"], num_hiddens=512, num_heads=8),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=512, num_hiddens=256, num_heads=4),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=256, num_hiddens=128, num_heads=2),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=128, num_hiddens=64, num_heads=2),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=64, num_hiddens=kwargs["output_size"], num_heads=1,  isMid=False),
            nn.PReLU()
        )
        for name,param in self.s.named_parameters():
            param.data = param.data.double()

    def make_ortho_weights(self,x):
        eps= 1e-7
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_sym = torch.mm(x.t(), x) 
        x_sym += torch.eye(x_sym.shape[1],device=device)*eps
        L = torch.linalg.cholesky(x_sym)
        ortho_weights = ((x.shape[0])**(1/2) )* (torch.inverse(L)).t()
        return ortho_weights
    
    def ortho_update(self,x):
        ortho_weights = self.make_ortho_weights(x)
        self.W_ortho = ortho_weights
        
    def forward(self, x, ortho_step=False):
        # x_net = normalize(self.s(x), dim=1)
        x_net = self.s(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho               
        return y

class AttentionNetText(nn.Module):
    '''
        四层注意力的纯注意力网络。
        add by lwb
        date: 2022-09-11
    '''
    def __init__(self, **kwargs):
        super(AttentionNetText, self).__init__()
        self.s = nn.Sequential(
            MultiHeadAttention(embed_dim=kwargs["input_size"], num_hiddens=256, num_heads=8),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=256, num_hiddens=256, num_heads=4),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=256, num_hiddens=128, num_heads=4),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=128, num_hiddens=64, num_heads=2),
            nn.ReLU(),
            MultiHeadAttention(embed_dim=64, num_hiddens=kwargs["output_size"], num_heads=1, isMid=False),
            nn.Tanh()
        )
        for name,param in self.s.named_parameters():
            param.data = param.data.double()

    def make_ortho_weights(self,x):
        eps= 1e-7
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_sym = torch.mm(x.t(), x) 
        x_sym += torch.eye(x_sym.shape[1],device=device)*eps
        L = torch.linalg.cholesky(x_sym)
        ortho_weights = ((x.shape[0])**(1/2) )* (torch.inverse(L)).t()
        return ortho_weights
    
    def ortho_update(self,x):
        ortho_weights = self.make_ortho_weights(x)
        self.W_ortho = ortho_weights
        
    def forward(self, x, ortho_step=False):
        # x_net = normalize(self.s(x), dim=1)
        x_net = self.s(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho               
        return y
