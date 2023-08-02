from torch import nn

class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE,self).__init__()
        self.encoder = nn.Sequential( 
            nn.Linear(kwargs["input_size"],1024), 
             nn.ReLU(),
             nn.Linear(1024,512), 
             nn.ReLU(),
             nn.Linear(512,kwargs["code_size"]))
            
        self.decoder = nn.Sequential( 
            nn.Linear(kwargs["code_size"],512), 
             nn.ReLU(),
             nn.Linear(512,1024), 
             nn.ReLU(),
             nn.Linear(1024,kwargs["input_size"]), 
             nn.Sigmoid())
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#专为AG_news 找到的一个内容.:https://github.com/ferjorosa/ag_news_pytorch/blob/main/pytorch_notebook.ipynb
class AE_TEXT(nn.Module):
    def __init__(self, vocab_size,embed_dim,**kwargs):
        super(AE_TEXT,self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.encoder = nn.Sequential( 
            nn.Linear(kwargs["input_size"],1024), 
             nn.ReLU(),
             nn.Linear(1024,512), 
             nn.ReLU(),
             nn.Linear(512,kwargs["code_size"]))
            
        self.decoder = nn.Sequential( 
            nn.Linear(kwargs["code_size"],512), 
             nn.ReLU(),
             nn.Linear(512,1024), 
             nn.ReLU(),
             nn.Linear(1024,kwargs["input_size"]), 
             nn.Sigmoid())
        
    def forward(self, x):
        embed = self.embedding(x)
        x = self.encoder(embed)
        x = self.decoder(x)
        return x,embed
