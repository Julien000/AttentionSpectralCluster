import logging
import os
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm 
from torch.nn import functional
from Modules.model.autoencoder import AE, AE_TEXT

def load_AE( test_dataloader, args,isText=False ,train_dataloader=None,dataName="MNIST"):
    filedir = dataName+".pkl"
    if os.path.exists(filedir):
        model_ae = torch.load(filedir)
    else:    
        logging.info("train AE")
        if isText:# 基于文本的AE
          model_ae = train_AE_text( train_dataloader,val_dataloader=test_dataloader,file=filedir, args=args)
        else:
          model_ae = train_AE( train_dataloader,val_dataloader=test_dataloader,file=filedir, args=args)
    return model_ae

def train_AE(train_dataloader,val_dataloader, file,args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AE(input_size= args.input_size,code_size=args.code_size).to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = nn.MSELoss()

  for epoch in range(args.AE_epochs):
      loss_t = 0
      model.train()
      for batch_features,tar in tqdm(train_dataloader,desc=f'AE_Epoch [{epoch+1}/{args.AE_epochs}]'):
          batch_features = batch_features[0].to(device)
          bs, c,h,w= batch_features.shape
          batch_features=batch_features.to(device)
          batch_features = batch_features.reshape(bs, c,-1).type(torch.float32)
          optimizer.zero_grad()
          
          outputs = model(batch_features)
          
          train_loss = criterion(outputs, batch_features)
          
          train_loss.backward()
          
          optimizer.step()
          
          loss_t += train_loss.item()

      loss_t = loss_t / len(train_dataloader)
      loss_v = 0
      model.eval()
      for batch_features,tar in val_dataloader:
          batch_features = batch_features.to(device)
          bs, c,h,w= batch_features.shape
          batch_features=batch_features.to(device)
          batch_features = batch_features.reshape(bs,c,-1).type(torch.float32)
          outputs = model(batch_features)
          val_loss = criterion(outputs, batch_features)
          loss_v += val_loss.item()

      loss_v = loss_v / len(val_dataloader)
      scheduler.step(val_loss)
      act_lr = optimizer.param_groups[0]['lr']  
      if args.verbose:
        print("epoch : {}/{}, learning_rate {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, args.AE_epochs,act_lr, loss_t, loss_v))
      if act_lr <= 1e-7:
        break
  if file!= None:
    dir = "output/pretrain/AE"
    if not os.path.exists(dir):
      os.makedirs(dir)
    torch.save(model, file)
  return model

def train_AE_text(train_dataloader,val_dataloader,file,args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AE(input_size= args.input_size,code_size=args.code_size).to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = nn.MSELoss()

  for epoch in range(args.AE_epochs):
      loss_t = 0
      model.train()
      for batch_features,_,tar in tqdm(train_dataloader,desc=f'AE_Epoch [{epoch+1}/{args.AE_epochs}]'):
          bs, f= batch_features.shape
          batch_features=batch_features.to(device)
          batch_features = batch_features.reshape(bs, 1,-1).type(torch.float32)
          optimizer.zero_grad()
          
          outputs = model(batch_features)
          
          train_loss = criterion(outputs, batch_features)
          
          train_loss.backward()
          
          optimizer.step()
          
          loss_t += train_loss.item()

      loss_t = loss_t / len(train_dataloader)
      loss_v = 0
      model.eval()
      for batch_features,tar in val_dataloader:
          bs, f= batch_features.shape
          batch_features=batch_features.to(device)
          batch_features = batch_features.reshape(bs, 1,-1).type(torch.float32)
          outputs = model(batch_features)
          val_loss = criterion(outputs, batch_features)
          loss_v += val_loss.item()

      loss_v = loss_v / len(val_dataloader)
      scheduler.step(val_loss)
      act_lr = optimizer.param_groups[0]['lr']  
      if args.verbose:
        print("epoch : {}/{}, learning_rate {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, args.AE_epochs,act_lr, loss_t, loss_v))
      if act_lr <= 1e-7:
        break
  if file!= None:
    dir = "output/pretrain/AE"
    if not os.path.exists(dir):
      os.makedirs(dir)
    torch.save(model, file)
  return model

 #根据使用的AG_news 数据集，我们用来做了第二个操作，将空白的东西给embedding 
def train_AE_text2(train_dataloader,val_dataloader,file,args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AE_TEXT(input_size= args.input_size,code_size=args.code_size).to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = nn.MSELoss()

  for epoch in range(args.AE_epochs):
      loss_t = 0
      model.train()
      for batch_features,_,tar in tqdm(train_dataloader,desc=f'AE_Epoch [{epoch+1}/{args.AE_epochs}]'):
          bs, f= batch_features.shape
          batch_features=batch_features.to(device)
          batch_features = batch_features.reshape(bs, 1,-1).type(torch.float32)
          optimizer.zero_grad()
          
          outputs,embed = model(batch_features)
          
          train_loss = criterion(outputs, embed)
          
          train_loss.backward()
          
          optimizer.step()
          
          loss_t += train_loss.item()

      loss_t = loss_t / len(train_dataloader)
      loss_v = 0
      model.eval()
      for batch_features,tar in val_dataloader:
          bs, f= batch_features.shape
          batch_features=batch_features.to(device)
          batch_features = batch_features.reshape(bs, 1,-1).type(torch.float32)
          outputs, embed = model(batch_features)
          val_loss = criterion(outputs, embed)
          loss_v += val_loss.item()

      loss_v = loss_v / len(val_dataloader)
      scheduler.step(val_loss)
      act_lr = optimizer.param_groups[0]['lr']  
      if args.verbose:
        print("epoch : {}/{}, learning_rate {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, args.AE_epochs,act_lr, loss_t, loss_v))
      if act_lr <= 1e-7:
        break
  if file!= None:
    dir = "output/pretrain/AE"
    if not os.path.exists(dir):
      os.makedirs(dir)
    torch.save(model, file)
  return model