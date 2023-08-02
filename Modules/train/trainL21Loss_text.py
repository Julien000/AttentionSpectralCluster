import logging
import torch.optim as optim
import torch
from tqdm import tqdm
from scipy.sparse import csgraph

from Modules.evaluation import evaluationForText
from Modules.loss import *
from Modules.evaluation import evalBasicRes_text
from Modules.model.spectralNet import SpectralNet, SpectralNetText
from Modules.model.spectralNet import SpectralNet_L21
from utils.basic import ChannelFussion, to_graph 
from  utils import exportExcel
from utils.data_contrainer import dataContainer, trainDataContainer
from Modules.model.attentionNet import *
import time

def train_SpectralNet_L21_text( train_dataloader,   test_dataloader,  model_ae,  args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = SpectralNetText(input_size=args.code_size, output_size=args.n_clusters).to(device)
  # model = SpectralNet_L21(input_size=args.code_size, output_size=args.n_clusters).to(device)
  criterion = L21LossV2(beta=args.beta)
  optimizer = optim.Adam(model.parameters(), lr=args.lr) 
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  train_length = len(train_dataloader)
  logging.info("================MODEL INFOR==============")
  logging.info(model)
  # 模型训练阶段
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  epochTimes=0
  for epoch in range(args.epochs):
    loss_t = 0
    epochTimes+=1
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length):
        x1, x2 = zipdata[0], zipdata[1]
        bs, c = x1.shape
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
        L=csgraph.laplacian(W ,normed=False)
        L=torch.from_numpy(L).to(device).double()
        W = torch.from_numpy(W).to(device).double() 
        Y = model(x1)        
        train_loss = criterion(Y,W, L)
        D=torch.diag(1/2*(torch.norm(Y,p=2,dim=1)))
        if args.isDivW:
          grads=((torch.mm(L,Y)+torch.mm(L.T,Y)))/(W.shape[0])+  args.beta*torch.mm(D,Y) 
        else:
          grads=(torch.mm(L,Y)+torch.mm(L.T,Y)+args.beta*torch.mm(D,Y))
        #更新梯度
        Y.backward(gradient=grads)
        optimizer.step()
        loss_t += train_loss.item()
    loss_t = loss_t / len(train_dataloader)
    # scheduler.step(loss_t)
    act_lr = optimizer.param_groups[0]['lr']
    #test
    # 模型评估部分。
    model.eval()
    evaluationForText(test_dataloader=test_dataloader, model=model, model_ae=model_ae, datacont=datacont, args=args, isold=True)
    # 记录结果部分
    test_size = len(test_dataloader)
    test_ACC_t ,test_NMI_t, test_ARI_t ,test_Purity_t = datacont.GetOneEpochResult(size=test_size)
    logging.info(f"epoch : {epoch + 1}/{args.epochs},ACC={test_ACC_t},NMI={test_NMI_t},\
        ARI={test_ARI_t},Purity={test_Purity_t}  ,learning_rate ={act_lr}, train_loss = {round(loss_t,4)}")
    
    if act_lr <= 1e-7:
      break
  
  # 保存模型，最后结果查看部分
  if args.file!= None:
    torch.save(model, args.file)
  #TODO: 6、导出结果部分
  cols=('model','epoch', "datasets",'ACC', 'NMI','ARI', 'purity', 'AVG_ACC', "AVG_NMI","AVG_ARI", "AVG_purity","n_clusters",'learningRate','batch_size')
  [max_ACC, max_NMI, max_ARI, max_purity] = datacont.getMax()
  [avg_ACC, avg_NMI, avg_ARI, avg_purity] = datacont.getAvg(times=epochTimes)
  model_type=args.model_type+"_"+args.data
  rows = [model_type, args.epochs, args.data, max_ACC, max_NMI, max_ARI, max_purity, avg_ACC, avg_NMI, avg_ARI, avg_purity,args.n_clusters,args.lr,args.batch_size] 
  logging.info(f"======{model_type}-Resule ACC={max_ACC},NMI={max_NMI},ARI={max_ARI},Purity={max_purity} ")
  # 保存模型
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows

def train_SpectralNet_attNet_text( train_dataloader, test_dataloader, model_ae, args):
  """
    纯注意力的网络
  """
  beta = args.beta
  model_type=args.model_type+"_"+args.data
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AttentionNetText(input_size=args.code_size, output_size=args.n_clusters).to(device)
  logging.info("================MODEL INFOR==============")
  logging.info(model)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = L21LossV2(beta=args.beta)
  # criterion = L0Loss(beta=args.beta)
  # criterion = NoNormLoss(beta=args.beta)
  # criterion = L1LossV2(beta=args.beta)
  train_length = len(train_dataloader)
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  datacont.ResetALL()
  trainData = trainDataContainer(epoch=args.epochs, len=4*3) 
  test_length = len(test_dataloader)
  epochTimes=0
  csvRows = []
  csvRows.append([args.model_type,'epoch', 'train_ACC', 'eval_ACC', 'train_loss', 'val_loss', 'Runtime'])
  for epoch in range(args.epochs):
    time_start = time.time()  # 记录开始时间
    trainData.ResetALL()
    loss_t = 0
    epochTimes+=1
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length) :
        x1, x2 = zipdata[0], zipdata[1]
        bs, c= x1.shape
        x1 = x1.reshape(bs,1,-1).float()
        x2 = x2.reshape(bs,1,-1).float()
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
        bs_sample = x1.reshape(bs,-1).detach().to("cpu").numpy()
        W = to_graph(bs_sample,"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        train_loss ,grads= criterion(Y,W)
        #更新梯度
        ACC, NMI, ARI, P = evalBasicRes_text( Y.detach().cpu().numpy(), zipdata[2].numpy(), args.n_clusters)
        trainData.setdata(data=[ACC,NMI,ARI,P])
        Y.backward(gradient=grads)
        optimizer.step()
        loss_t += torch.abs(torch.sum(grads)/W.shape[0]).item()

    #test
    act_lr = optimizer.param_groups[0]['lr']
    #test
    model.eval()
    val_losst = 0
    #模型验证部分
    for  x, tar in test_dataloader:
      bs, c= x.shape
      x= x.to(device)
      #embedding
      if args.embedding:
          x =x.reshape( bs,1, -1) # 对注意力增加通道
          x =  model_ae.encoder(x.float()).double()
      Y = model(x)
      W = to_graph(x.reshape(bs,-1).detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
      val_loss, grad = criterion(Y,W)
      val_losst += torch.abs(torch.sum(grad)/W.shape[0]).item()
      Y=Y.reshape(bs,-1)
      ACC, NMI, ARI, P = evalBasicRes_text( Y.detach().cpu().numpy(), tar.numpy(), args.n_clusters)
      datacont.setdata(data=[ACC,NMI,ARI,P])
    # 记录结果部分
    # if act_lr <= 1e-7:
    #   break
    # 结果处理 部分

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    loss_t = loss_t / train_length
    val_losst = val_losst/ test_length
    valACC ,valNMI, valARI, valP = datacont.GetOneEpochResult(size=test_length)
    [*trainRows] = trainData.getdata(size=train_length)
    newrow=[' ', epoch, trainRows[0], valACC, loss_t, val_losst, time_sum]
    csvRows.append(newrow)
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
  testACC, testNMI,testARI, testP = evaluationForText(test_dataloader=test_dataloader, model=model, model_ae=model_ae, args=args, isold=True)
  csvRows.append(["Text_Test",testACC,testNMI, testARI])
  logging.info(f"======{model_type}--TestResule-Resule ACC={testACC},NMI={testNMI},ARI={testARI},Purity={testP} ")
  
  #TODO: 6、导出训练结果
  exportExcel.export_csv(path=args.traindir+args.model_type+args.log_fileName+".csv", rowList=csvRows)
 
  # 保存模型
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows

 
