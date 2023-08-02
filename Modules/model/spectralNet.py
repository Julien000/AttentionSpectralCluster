import torch
from torch import nn
from torch.nn.functional import normalize


class SpectralNet(nn.Module):
    '''
        原始的spectralNET网络模型。
    '''
    def __init__(self, **kwargs):
        super(SpectralNet, self).__init__()
        self.s = nn.Sequential( 
            nn.Linear(kwargs["input_size"],1024), 
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,kwargs["output_size"]),
            nn.Tanh())
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
        x_net = self.s(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho               
        return y
class SpectralNetText(nn.Module):
    '''
        原始的spectralNET网络模型。
    '''
    def __init__(self, **kwargs):
        super(SpectralNetText, self).__init__()
        self.s = nn.Sequential( 
            nn.Linear(kwargs["input_size"],512), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,kwargs["output_size"]),
            nn.Tanh())
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
        x_net = self.s(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho               
        return y

class SpectralNet_L21(nn.Module):
    def __init__(self, **kwargs):
        super(SpectralNet_L21, self).__init__()
        self.s = nn.Sequential( 
                nn.Linear(kwargs["input_size"],1024), 
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512,kwargs["output_size"]),
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
#SpectralNet_L21BM(
#   (s): Sequential(
#     (0): Linear(in_features=10, out_features=1024, bias=True)
#     (1): Linear(in_features=1024, out_features=2048, bias=True)
#     (2): ReLU()
#     (3): Linear(in_features=2048, out_features=1024, bias=True)
#     (4): ReLU()
#     (5): Linear(in_features=1024, out_features=512, bias=True)
#     (6): ReLU()
#     (7): Linear(in_features=512, out_features=10, bias=True)
#     (8): PReLU(num_parameters=1)
#   )
# )
# 因为加了batch norm 就不再需要变成double类型了
class Spectral_L1_rule(nn.Module):
    def __init__(self, **kwargs):
        super(Spectral_L1_rule, self).__init__()
        self.s = nn.Sequential( 
                nn.Linear(kwargs["input_size"],1024), 
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512,kwargs["output_size"]),
                nn.ReLU()
            )
        
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
class Spectral_L1_tanh(nn.Module):
    def __init__(self, **kwargs):
        super(Spectral_L1_tanh, self).__init__()
        self.s = nn.Sequential( 
                nn.Linear(kwargs["input_size"],1024), 
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512,kwargs["output_size"]),
                nn.Tanh()
            )
        
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
class Spectral_L1_Prule(nn.Module):
    def __init__(self, **kwargs):
        super(Spectral_L1_Prule, self).__init__()
        self.s = nn.Sequential( 
                nn.Linear(kwargs["input_size"],1024), 
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512,kwargs["output_size"]),
                nn.PReLU()
            )
        
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
class SpectralNet_L21BM(nn.Module):
    def __init__(self, **kwargs):
        super(SpectralNet_L21BM, self).__init__()
        self.s = nn.Sequential( 
                nn.Linear(kwargs["input_size"],1024), 
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Linear(2048,1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024,kwargs["output_size"]),
                # nn.PReLU()
                nn.ReLU()
                )
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


class SpectralNet_L1BM(nn.Module):
    def __init__(self, **kwargs):
        super(SpectralNet_L1BM, self).__init__()
        self.s = nn.Sequential( 
                nn.Linear(kwargs["input_size"],1024), 
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048,momentum=0.9),
                # nn.InstanceNorm1d(2048),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024,momentum=0.9),
                # nn.InstanceNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512,momentum=0.9),
                # nn.InstanceNorm1d(512),
                nn.Linear(512,kwargs["output_size"]),
                nn.PReLU()
                )


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
        # x_net = normalize(self.s(x), dim=2)
        x_net = self.s(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho               
        return y