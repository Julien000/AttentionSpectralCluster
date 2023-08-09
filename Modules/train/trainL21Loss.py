import logging
import os
import torch.optim as optim
import torch
from tqdm import tqdm
from scipy.sparse import csgraph

from Modules.evaluation import evalBasicRes, evaluationForImage
from Modules.loss import *
from Modules.model.spectralNet import SpectralNet
from Modules.model.spectralNet import SpectralNet_L21
from Modules.similarity import *
from utils.basic import ChannelFussion, to_graph 
from  utils import exportExcel
from utils.data_contrainer import dataContainer, trainDataContainer
from Modules.model.attentionNet import *
from visdom import Visdom
import time

def train_SpectralNet_L21( train_dataloader,   test_dataloader,  model_ae,  args, model_siam=None):
  model_type=args.model_type+"_"+args.data
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = SpectralNet_L21(input_size=args.code_size, output_size=args.n_clusters).to(device)
  # criterion = L21LossV2(beta=args.beta)
  criterion = L1LossV2(beta=args.beta)
  optimizer = optim.Adam(model.parameters(), lr=args.lr) 
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  logging.info("================MODEL INFOR==============")
  logging.info(model)
  # 获取数据集单个epochsize
  train_length = len(train_dataloader)
  test_length = len(test_dataloader)

  # 创建数据容器
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  epochTimes=0 # 用于控制提前结束的结果平均
  trainData = trainDataContainer(epoch=args.epochs, len=4*3) 
 
  #训练结果list
  csvRows = []
  csvRows.append([args.model_type,'epoch', 'train_ACC', 'val_ACC'])
  # 实例化窗口
  wind_loss = Visdom()
  wind_acc = Visdom()
  # 初始化窗口参数
  wind_loss.line([[0.,0.]],[0.],win = 'FCL21_train_loss',opts = dict(title = 'FCL21_train_loss',legend = ['train_loss', 'val_loss']))
  wind_acc.line([[0.,0.]],[0.],win = 'FCL21_ACC',opts = dict(title = 'FCL21_train&valACC',legend = ['train_acc','valACC']))

  # 模型训练阶段
  for epoch in range(args.epochs):
    loss_t = 0
    epochTimes+=1
    newrow=[]
    trainData.ResetALL()
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length):
        x1, x2 = zipdata[0][0], zipdata[0][1]
        bs, c, w, h = x1.shape
        x1 = x1.reshape(bs,-1).float()
        x2 = x2.reshape(bs,-1).float()
        if args.embedding:
          x1=model_ae.encoder(x1.to(device))
          x2=model_ae.encoder(x2.to(device))
        x1 = x1.double()
        x2 = x2.double()
        #ortostep
        model.eval()
        _ = model(x2,ortho_step=True)
        #gradstep
        model.train()
        optimizer.zero_grad()
        # if args.siam_metric:
        # #   x1 = model_siam(x1)
        W = to_graph(x1.detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        # W = getRatiocut(x1.detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        # # W = getmaxcut(x1.detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        # # W = to_graph(x1.detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        # W = to_graphWithT(x1.detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        # W = ncut(x1,tau = args.tau, eps=1e-7).cpu().numpy()
        # # W = getCosine_similarity(x1,tau = args.tau, eps=1e-7).detach().cpu().numpy()

        Y = model( x1)        
        train_loss,grads= criterion(Y,W)
        #更新梯度
        # train_loss.backward()
        Y.backward(gradient=grads)
        optimizer.step()
        loss_t += train_loss.item()
        #记录训练结果
        ACC, NMI, ARI, P = evalBasicRes( Y, zipdata[1].numpy(), args.n_clusters)
        trainData.setdata(data=[ACC,NMI,ARI,P])

    model.eval() 
    val_losst = 0
    #模型验证部分
    for  x, tar in test_dataloader:
      bs, c, w, h = x.shape
      x= x.to(device)
      #embedding
      if args.embedding:
          if(c>1):
              x = x.reshape( bs, c, -1).to(device)
          else:
              x =x.reshape( bs, -1).to(device)
          if False: # 对注意力增加通道
            x =x.reshape( bs,1, -1)
          x =  model_ae.encoder(x.float()).double()
      Y = model(x)
      W = to_graph(x.detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
      val_loss, grad = criterion(Y,W)
      val_losst += val_loss.item()
      Y=Y.reshape(bs,-1)
      ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
      datacont.setdata(data=[ACC,NMI,ARI,P])

    loss_t = loss_t / train_length
    val_losst = val_losst/ test_length
    valACC ,valNMI, valARI, valP = datacont.GetOneEpochResult(size=test_length)
    [*trainRows] = trainData.getdata(size=train_length)
    newrow=[' ',epoch, trainRows[0], valACC]
    csvRows.append(newrow)
    wind_loss.line([[round(loss_t,4),round(val_losst,4)]],[epoch],win = 'FCL21_train_loss',update = 'append')
    wind_acc.line([[trainRows[0],valACC]],[epoch],win = 'FCL21_ACC',update = 'append')
    time.sleep(0.5)
    # 记录结果部分
    # 学习率终止策略
    scheduler.step(val_loss)
    act_lr = optimizer.param_groups[0]['lr']
    logging.info(f"epoch : {epoch + 1}/{args.epochs},trainACC={trainRows[0]},ACC={valACC},NMI={valNMI}, ARI={valARI},Purity={valP}  ,learning_rate ={act_lr}, train_loss = {round(loss_t,4)}")
    if act_lr <= 1e-7:
      break
  # 模型测试部分
  [max_ACC, max_NMI, max_ARI, max_purity] = datacont.getMax()
  [avg_ACC, avg_NMI, avg_ARI, avg_purity] = datacont.getAvg(times=epochTimes)  
  rows = [model_type, args.epochs, args.data, max_ACC, max_NMI, max_ARI, max_purity, avg_ACC, avg_NMI, avg_ARI, avg_purity,args.n_clusters,args.lr,args.batch_size] 
  datacont.ResetALL()
  
  model.eval()
  testACC, testNMI,testARI, testP = evaluationForImage(test_dataloader=test_dataloader, model=model, model_ae=model_ae, args=args, isold=True)
  csvRows.append(["Test",testACC,testNMI, testARI])
  logging.info(f"======{model_type}-- TestResule-Resule ACC={testACC},NMI={testNMI},ARI={testARI},Purity={testP} ")
  #TODO: 6、导出训练结果
  exportExcel.export_csv(path=args.traindir+args.model_type+".csv", rowList=csvRows)
  # 保存模型
  dir = "output/pretrain/NET"
  if not os.path.exists(dir):
      os.makedirs(dir)
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows

def train_SpectralNet_att1( train_dataloader, test_dataloader, model_ae, args, model_siam=None):
  """
    训练单层的attentin
  """
  model_type=args.model_type+"_"+args.data
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AttentionNet1(input_size=args.code_size, output_size=args.n_clusters).to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = L21LossV2(beta=args.beta)
  train_length = len(train_dataloader)
  test_length = len(test_dataloader)
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  logging.info("================MODEL INFOR==============")
  logging.info(model)
  epochTimes=0
  trainData = trainDataContainer(epoch=args.epochs, len=4*3) 
  csvRows = []
  csvRows.append([args.model_type,'epoch', 'train_ACC', 'eval_ACC'])
  # 实例化窗口
  wind_loss = Visdom()
  wind_acc = Visdom()
  # 初始化窗口参数
  wind_loss.line([[0.,0.]],[0.],win = 'A1_train_loss',opts = dict(title = 'A1_train_loss',legend = ['train_loss', 'val_loss']))
  wind_acc.line([[0.,0.]],[0.],win = 'A1_ACC',opts = dict(title = 'A1_train&valACC',legend = ['train_acc','valACC']))

  for epoch in range(args.epochs):
    trainData.ResetALL()
    loss_t = 0
    epochTimes+=1
    newrow=[]
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length) :
        x1, x2 = zipdata[0][0], zipdata[0][1]
        bs, c, w, h = x1.shape
        # 对于注意力一定要包含 c 的size 
        x1 = x1.reshape(bs,c,-1).float()
        x2 = x2.reshape(bs,c,-1).float()
        # embedding
        if args.embedding:
          x1=model_ae.encoder(x1.to(device))
          x2=model_ae.encoder(x2.to(device))
        x2=x2.double()
        x1=x1.double()
        #ortostep
        model.eval()
        _ = model(x2,ortho_step=True)

        #gradstep
        model.train()
        optimizer.zero_grad()
        Y = model(x1)
        if c>1:
          Y = ChannelFussion(Y)
          x1 = ChannelFussion(x1) 
        if args.siam_metric:
          x1 = model_siam(x1)
        bs_sample = x1.reshape(bs,-1).detach().to("cpu").numpy()
        W = to_graph(bs_sample,"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()     
        train_loss ,grads= criterion(Y,W)
        #更新梯度
        ACC, NMI, ARI, P = evalBasicRes( Y, zipdata[1].numpy(), args.n_clusters)
        trainData.setdata(data=[ACC,NMI,ARI,P])
        Y.backward(gradient=grads)
        optimizer.step()
        loss_t += train_loss.item()
    model.eval() 
    val_losst = 0
    #模型验证部分
    for  x, tar in test_dataloader:
      bs, c, w, h = x.shape
      x= x.to(device)
      #embedding
      if args.embedding:
          if(c>1):
            x = x.reshape( bs, c, -1).to(device)
          else:
            x =x.reshape( bs,1, -1) # 对注意力增加通道
          x =  model_ae.encoder(x.float()).double()
      Y = model(x)
      W = to_graph(x.reshape(bs,-1).detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
      val_loss, grad = criterion(Y,W)
      val_losst += val_loss.item()
      Y=Y.reshape(bs,-1)
      ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
      datacont.setdata(data=[ACC,NMI,ARI,P])

    loss_t = loss_t / train_length
    val_losst = val_losst/ test_length
    valACC ,valNMI, valARI, valP = datacont.GetOneEpochResult(size=test_length)
    [*trainRows] = trainData.getdata(size=train_length)

    newrow=[' ',epoch, trainRows[0], valACC]
    csvRows.append(newrow)
    wind_loss.line([[round(loss_t,4),round(val_losst,4)]],[epoch],win = 'A1_train_loss',update = 'append')
    wind_acc.line([[trainRows[0],valACC]],[epoch],win = 'A1_ACC',update = 'append')
    time.sleep(0.5)
    # 记录结果部分
    # 学习率终止策略
    scheduler.step(val_loss)
    act_lr = optimizer.param_groups[0]['lr']
    logging.info(f"epoch : {epoch + 1}/{args.epochs},trainACC={trainRows[0]},ACC={valACC},NMI={valNMI}, ARI={valARI},Purity={valP}  ,learning_rate ={act_lr}, train_loss = {round(loss_t,4)}")
    if act_lr <= 1e-7:
      break
  # 模型测试部分
  [max_ACC, max_NMI, max_ARI, max_purity] = datacont.getMax()
  [avg_ACC, avg_NMI, avg_ARI, avg_purity] = datacont.getAvg(times=epochTimes)  
  rows = [model_type, args.epochs, args.data, max_ACC, max_NMI, max_ARI, max_purity, avg_ACC, avg_NMI, avg_ARI, avg_purity,args.n_clusters,args.lr,args.batch_size] 
  datacont.ResetALL()
  
  model.eval()
  testACC, testNMI,testARI, testP = evaluationForImage(test_dataloader=test_dataloader, model=model, model_ae=model_ae, args=args, isold=True)
  csvRows.append(["Test",testACC,testNMI, testARI])
  logging.info(f"======{model_type}-- TestResule-Resule ACC={testACC},NMI={testNMI},ARI={testARI},Purity={testP} ")
  #TODO: 6、导出训练结果
  exportExcel.export_csv(path=args.traindir+args.model_type+".csv", rowList=csvRows)
  # 保存模型
  dir = "output/pretrain/NET"
  if not os.path.exists(dir):
      os.makedirs(dir)
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows

def train_SpectralNet_att3( train_dataloader, test_dataloader, model_ae, args, model_siam=None):
  """
    根据老师的结果进行复现，效果最好的一个网络
  """
  model_type=args.model_type+"_"+args.data
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AttentionNet3(input_size=args.code_size, output_size=args.n_clusters).to(device)
  logging.info("================MODEL INFOR==============")
  logging.info(model)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = L21LossV2(beta=args.beta)
  train_length = len(train_dataloader)
  test_length = len(test_dataloader)
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  epochTimes=0
  trainData = trainDataContainer(epoch=args.epochs, len=4*3) 
  csvRows = []
  csvRows.append([args.model_type,'epoch', 'train_ACC', 'eval_ACC'])
  # 实例化窗口
  wind_loss = Visdom()
  # 初始化窗口参数
  # loss, ACC, epoch
  wind_loss.line([[0.,0.]],[0.],win = 'A3_train_loss',opts = dict(title = 'A3_train_loss',legend = ['train_loss', 'val_loss']))
  # 实例化窗口
  wind_acc = Visdom()
  # 初始化窗口参数
  wind_acc.line([[0.,0.]],[0.],win = 'A3_ACC',opts = dict(title = 'A3_train&valACC',legend = ['train_acc','valACC']))

  for epoch in range(args.epochs):
    trainData.ResetALL()
    loss_t = 0
    epochTimes+=1
    newrow=[]
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length) :
        x1, x2 = zipdata[0][0], zipdata[0][1]
        bs, c, w, h = x1.shape
        x1 = x1.reshape(bs,c,-1).float()
        x2 = x2.reshape(bs,c,-1).float()
        # embedding
        if args.embedding:
          x1=model_ae.encoder(x1.to(device))
          x2=model_ae.encoder(x2.to(device))
        x2=x2.double()
        x1=x1.double()
        #ortostep
        model.eval()
        _ = model(x2,ortho_step=True)
        #gradstep
        model.train()
        optimizer.zero_grad()
        Y = model(x1)
        if c>1:
          Y = ChannelFussion(Y)
          x1 = ChannelFussion(x1) 
        if args.siam_metric:
          x1 = model_siam(x1)
        bs_sample = x1.reshape(bs,-1).detach().to("cpu").numpy()
        W = to_graph(bs_sample,"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        train_loss ,grads= criterion(Y,W)
        #更新梯度
        ACC, NMI, ARI, P = evalBasicRes( Y, zipdata[1].numpy(), args.n_clusters)
        trainData.setdata(data=[ACC,NMI,ARI,P])
        Y.backward(gradient=grads)
        optimizer.step()
        loss_t += train_loss.item()
    model.eval() 
    val_losst = 0
    #模型验证部分
    for  x, tar in test_dataloader:
      bs, c, w, h = x.shape
      x= x.to(device)
      #embedding
      if args.embedding:
          if(c>1):
            x = x.reshape( bs, c, -1).to(device)
          else:
            x =x.reshape( bs,1, -1) # 对注意力增加通道
          x =  model_ae.encoder(x.float()).double()
      Y = model(x)
      W = to_graph(x.reshape(bs,-1).detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
      val_loss, grad = criterion(Y,W)
      val_losst += val_loss.item()
      Y=Y.reshape(bs,-1)
      ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
      datacont.setdata(data=[ACC,NMI,ARI,P])

    loss_t = loss_t / train_length
    val_losst = val_losst/ test_length
    valACC ,valNMI, valARI, valP = datacont.GetOneEpochResult(size=test_length)
    [*trainRows] = trainData.getdata(size=train_length)
    newrow=[' ',epoch, trainRows[0], valACC]
    csvRows.append(newrow)
    wind_loss.line([[round(loss_t,4),round(val_losst,4)]],[epoch],win = 'A3_train_loss',update = 'append')
    wind_acc.line([[trainRows[0],valACC]],[epoch],win = 'A3_ACC',update = 'append')
    time.sleep(0.5)
    # 记录结果部分
    # 学习率终止策略
    scheduler.step(val_loss)
    act_lr = optimizer.param_groups[0]['lr']
    logging.info(f"epoch : {epoch + 1}/{args.epochs},trainACC={trainRows[0]},ACC={valACC},NMI={valNMI}, ARI={valARI},Purity={valP}  ,learning_rate ={act_lr}, train_loss = {round(loss_t,4)}")
    if act_lr <= 1e-7:
      break
  
  # 保存单词计算的结果
  [max_ACC, max_NMI, max_ARI, max_purity] = datacont.getMax()
  [avg_ACC, avg_NMI, avg_ARI, avg_purity] = datacont.getAvg(times=epochTimes)  
  rows = [model_type, args.epochs, args.data, max_ACC, max_NMI, max_ARI, max_purity, avg_ACC, avg_NMI, avg_ARI, avg_purity,args.n_clusters,args.lr,args.batch_size] 
  datacont.ResetALL()
  
  # 模型测试部分
  model.eval()
  testACC, testNMI,testARI, testP = evaluationForImage(test_dataloader=test_dataloader, model=model, model_ae=model_ae, args=args, isold=True)
  csvRows.append(["Test",testACC,testNMI, testARI])
  logging.info(f"======{model_type}-- TestResule-Resule ACC={testACC},NMI={testNMI},ARI={testARI},Purity={testP} ")
  #TODO: 6、导出训练结果
  exportExcel.export_csv(path=args.traindir+args.model_type+".csv", rowList=csvRows)
  # 保存模型
  dir = "output/pretrain/NET"
  if not os.path.exists(dir):
      os.makedirs(dir)
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows

def train_SpectralNet_attNet( train_dataloader, test_dataloader, model_ae, args, model_siam=None):
  """
    纯注意力的网络
  """
  model_type=args.model_type+"_"+args.data
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AttentionNet(input_size=args.code_size, output_size=args.n_clusters).to(device)
  logging.info("================MODEL INFOR==============")
  logging.info(model)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = L21LossV2(beta=args.beta)
  # criterion = L21Loss_D(beta=args.beta)
  # criterion = L1LossV2(beta=args.beta)
  # criterion = NoNormLoss(beta=args.beta)
  # criterion = L0Loss(beta=args.beta)

  train_length = len(train_dataloader)
  test_length = len(test_dataloader)
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  datacont.ResetALL()
  epochTimes=0
  trainData = trainDataContainer(epoch=args.epochs, len=4*3) 
  csvRows = []
  csvRows.append([args.model_type,'epoch', 'train_ACC', 'eval_ACC', 'train_loss', 'val_loss', 'runTime'])

  # 实例化窗口
  wind_loss = Visdom()
  # 初始化窗口参数
  # loss, ACC, epoch
  wind_loss.line([[0.,0.]],[0.],win = 'ANet_train_loss',opts = dict(title = 'ANet_train_loss',legend = ['train_loss', 'val_loss']))
  # 实例化窗口
  wind_acc = Visdom()
  # 初始化窗口参数
  wind_acc.line([[0.,0.]],[0.],win = 'ANet_ACC',opts = dict(title = 'ANet_train&valACC',legend = ['train_acc','valACC']))

  for epoch in range(args.epochs):
    time_start = time.time()  # 记录开始时间
    trainData.ResetALL()
    loss_t = 0
    epochTimes+=1
    newrow=[]
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length) :
        x1, x2 = zipdata[0][0], zipdata[0][1]
        bs, c, w, h = x1.shape
        # 对于注意力一定要包含 c 的size 
        x1 = x1.reshape(bs,c,-1).float()
        x2 = x2.reshape(bs,c,-1).float()
        # embedding
        if args.embedding:
          x1=model_ae.encoder(x1.to(device))
          x2=model_ae.encoder(x2.to(device))
        x2=x2.double()
        x1=x1.double()
        #ortostep
        model.eval()
        _ = model(x2,ortho_step=True)

        #gradstep
        model.train()
        optimizer.zero_grad()
        Y = model(x1)
        if c>1:
          Y = ChannelFussion(Y)
          x1 = ChannelFussion(x1) 
        if args.siam_metric:
          x1 = model_siam(x1)
        bs_sample = x1.reshape(bs,-1).detach().to("cpu").numpy()
        W = to_graph(bs_sample,"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        train_loss ,grads= criterion(Y,W)
        #更新梯度
        ACC, NMI, ARI, P = evalBasicRes( Y, zipdata[1].numpy(), args.n_clusters)
        trainData.setdata(data=[ACC,NMI,ARI,P])
        Y.backward(gradient=grads)
        optimizer.step()
        loss_t += torch.abs(torch.sum(grads)/W.shape[0]).item()

    model.eval() 
    val_losst = 0
    #模型验证部分
    for  x, tar in test_dataloader:
      bs, c, w, h = x.shape
      x= x.to(device)
      #embedding
      if args.embedding:
          if(c>1):
            x = x.reshape( bs, c, -1).to(device)
          else:
            x =x.reshape( bs,1, -1) # 对注意力增加通道
          x =  model_ae.encoder(x.float()).double()
      Y = model(x)
      W = to_graph(x.reshape(bs,-1).detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
      val_loss, grad = criterion(Y,W)
      val_losst += torch.abs(torch.sum(grad)/W.shape[0]).item()
      Y=Y.reshape(bs,-1)
      ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
      datacont.setdata(data=[ACC,NMI,ARI,P])

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    loss_t = loss_t / train_length
    val_losst = val_losst/ test_length
    valACC ,valNMI, valARI, valP = datacont.GetOneEpochResult(size=test_length)
    [*trainRows] = trainData.getdata(size=train_length)
    newrow=[' ',epoch, trainRows[0], valACC, loss_t, val_losst, time_sum]
    csvRows.append(newrow)
    wind_loss.line([[round(loss_t,4),round(val_losst,4)]],[epoch],win = 'ANet_train_loss',update = 'append')
    wind_acc.line([[trainRows[0],valACC]],[epoch],win = 'ANet_ACC',update = 'append')
    time.sleep(0.5)
    # 记录结果部分
    # 学习率终止策略
    scheduler.step(val_loss)
    act_lr = optimizer.param_groups[0]['lr']
    logging.info(f"epoch : {epoch + 1}/{args.epochs},trainACC={trainRows[0]},ACC={valACC},NMI={valNMI}, ARI={valARI},Purity={valP}  ,learning_rate ={act_lr}, train_loss = {round(loss_t,4)}")
    # if act_lr <= 1e-7:
    #   break
  # 保存模型训练的结果
  [max_ACC, max_NMI, max_ARI, max_purity] = datacont.getMax()
  [avg_ACC, avg_NMI, avg_ARI, avg_purity] = datacont.getAvg(times=epochTimes)  
  rows = [model_type, args.epochs, args.data, max_ACC, max_NMI, max_ARI, max_purity, avg_ACC, avg_NMI, avg_ARI, avg_purity,args.n_clusters,args.lr,args.batch_size] 
  datacont.ResetALL()
  # 模型测试部分
  model.eval()
  testACC, testNMI,testARI, testP = evaluationForImage(test_dataloader=test_dataloader, model=model, model_ae=model_ae, args=args, isold=True)
  csvRows.append(["Test",testACC,testNMI, testARI])
  logging.info(f"======{model_type}-- TestResule-Resule ACC={testACC},NMI={testNMI},ARI={testARI},Purity={testP} ")
  
  #TODO: 6、导出训练结果
  exportExcel.export_csv(path=args.traindir+args.model_type+args.log_fileName+".csv", rowList=csvRows)
 
  # 保存模型
  dir = "output/pretrain/NET"
  if not os.path.exists(dir):
      os.makedirs(dir)
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows

def train_SpectralNet_Test( train_dataloader,   test_dataloader,  model_ae,  args, model_siam=None):
  model_type="TEST_"+args.data
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = SpectralNet_L21(input_size=args.code_size, output_size=args.n_clusters).to(device)
  # criterion = L21LossV2(beta=args.beta)
  criterion = BlockDiagLossW() # 基于W 权重矩阵的结果, 
  # criterion = BlockDiagLoss() # 基于样本计算的结果
  optimizer = optim.Adam(model.parameters(), lr=args.lr) 
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  logging.info("================MODEL INFOR==============")
  logging.info(model)
  # 模型训练阶段
  train_length = len(train_dataloader)
  test_length = len(test_dataloader)
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  epochTimes=0
  trainData = trainDataContainer(epoch=args.epochs, len=4*3) 
  csvRows = []
  csvRows.append([args.model_type,'epoch', 'train_ACC', 'val_ACC'])
  # 实例化窗口
  wind_loss = Visdom()
  wind_acc = Visdom()
  # 初始化窗口参数
  wind_loss.line([[0.,0.]],[0.],win = 'Test_train_loss',opts = dict(title = 'Test_train_loss',legend = ['train_loss', 'val_loss']))
  wind_acc.line([[0.,0.]],[0.],win = 'Test_ACC',opts = dict(title = 'Test_train&valACC',legend = ['train_acc','valACC']))

  for epoch in range(args.epochs):
    loss_t = 0
    epochTimes+=1
    newrow=[]
    trainData.ResetALL()
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length):
        x1, x2 = zipdata[0][0], zipdata[0][1]
        bs, c, w, h = x1.shape
        x1 = x1.reshape(bs,-1).float()
        x2 = x2.reshape(bs,-1).float()
        if args.embedding:
          x1=model_ae.encoder(x1.to(device))
          x2=model_ae.encoder(x2.to(device))
        x1 = x1.double()
        x2 = x2.double()
        #ortostep
        model.eval()
        _ = model(x2,ortho_step=True)
        #gradstep
        model.train()
        optimizer.zero_grad()
        W = to_graph(x1.detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        W = torch.from_numpy(W).to(device).double()      
        Y = model( x1)        
        train_loss,grads= criterion(Y,W)
        #更新梯度
        train_loss.backward()
        # Y.backward(gradient=grads)
        optimizer.step()
        loss_t += train_loss.item()
        #记录训练结果
        ACC, NMI, ARI, P = evalBasicRes( Y, zipdata[1].numpy(), args.n_clusters)
        trainData.setdata(data=[ACC,NMI,ARI,P])
    model.eval() 
    val_losst = 0
    #模型验证部分
    for  x, tar in test_dataloader:
      bs, c, w, h = x.shape
      x= x.to(device)
      #embedding
      if args.embedding:
          if(c>1):
              x = x.reshape( bs, c, -1).to(device)
          else:
              x =x.reshape( bs, -1).to(device)
          if False: # 对注意力增加通道
            x =x.reshape( bs,1, -1)
          x =  model_ae.encoder(x.float()).double()
      Y = model(x)
      W = to_graph(x.detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
      W = torch.from_numpy(W).to(device).double()      
      val_loss, grad = criterion(Y,W)
      val_losst += val_loss.item()
      Y=Y.reshape(bs,-1)
      ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
      datacont.setdata(data=[ACC,NMI,ARI,P])

    loss_t = loss_t / train_length
    val_losst = val_losst/ test_length
    valACC ,valNMI, valARI, valP = datacont.GetOneEpochResult(size=test_length)
    [*trainRows] = trainData.getdata(size=train_length)
    newrow=[' ',epoch, trainRows[0], valACC]
    csvRows.append(newrow)
    wind_loss.line([[round(loss_t,4),round(val_losst,4)]],[epoch],win = 'Test_train_loss',update = 'append')
    wind_acc.line([[trainRows[0],valACC]],[epoch],win = 'Test_ACC',update = 'append')
    time.sleep(0.5)
    # 记录结果部分
    # 学习率终止策略
    scheduler.step(val_loss)
    act_lr = optimizer.param_groups[0]['lr']
    logging.info(f"epoch : {epoch + 1}/{args.epochs},trainACC={trainRows[0]},ACC={valACC},NMI={valNMI}, ARI={valARI},Purity={valP}  ,learning_rate ={act_lr}, train_loss = {round(loss_t,4)}")
    if act_lr <= 1e-7:
      break
  # 模型测试部分
  [max_ACC, max_NMI, max_ARI, max_purity] = datacont.getMax()
  [avg_ACC, avg_NMI, avg_ARI, avg_purity] = datacont.getAvg(times=epochTimes)  
  rows = [model_type, args.epochs, args.data, max_ACC, max_NMI, max_ARI, max_purity, avg_ACC, avg_NMI, avg_ARI, avg_purity,args.n_clusters,args.lr,args.batch_size] 
  datacont.ResetALL()
  
  model.eval()
  testACC, testNMI,testARI, testP = evaluationForImage(test_dataloader=test_dataloader, model=model, model_ae=model_ae, args=args, isold=True)
  csvRows.append(["Test",testACC,testNMI, testARI])
  logging.info(f"======{model_type}-- TestResule-Resule ACC={testACC},NMI={testNMI},ARI={testARI},Purity={testP} ")
  #TODO: 6、导出训练结果
  exportExcel.export_csv(path=args.traindir+args.model_type+".csv", rowList=csvRows)
  # 保存模型
  dir = "output/pretrain/NET"
  if not os.path.exists(dir):
      os.makedirs(dir)
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows
