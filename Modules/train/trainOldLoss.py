
import logging
import torch
import torch.optim as optim
from tqdm import tqdm 
from Modules.evaluation import evalBasicRes, evaluationForImage
from Modules.loss import SpectralNetLoss
from Modules.model.attentionNet import AttentionNet3
from Modules.model.spectralNet import SpectralNet
from utils import exportExcel
from utils.basic import ChannelFussion, to_graph
from utils.data_contrainer import dataContainer, trainDataContainer
from visdom import Visdom
import time

def train_SpectralNet_oldLoss( train_dataloader, test_dataloader, model_ae, args, model_siam=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = SpectralNet(input_size=args.code_size, output_size=args.n_clusters).to(device)
  logging.info("================MODEL INFORMATION==============")
  logging.info(model)
  model_type=args.model_type+"_"+args.data
  test_length = len(test_dataloader)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = SpectralNetLoss()
  train_length = len(train_dataloader)
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  epochTimes=0
  trainData = trainDataContainer(epoch=args.epochs, len=4*3) 
  csvRows = []
  csvRows.append([args.model_type,'epoch', 'train_ACC', 'eval_ACC'])
    # 实例化窗口
  wind_loss = Visdom()
  wind_acc = Visdom()
  # 初始化窗口参数
  wind_loss.line([[0., 0.]],[0.],win = 'old_train_loss',opts = dict(title = 'old_train_loss',legend = ['train_loss', 'val_loss']))
  wind_acc.line([[0.,0.]],[0.],win = 'old_ACC',opts = dict(title = 'old_train&testACC',legend = ['train_acc','test_acc']))

  for epoch in range(args.epochs):
    loss_t = 0
    epochTimes+=1
    newrow=[]
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length) :
        trainData.ResetALL()
        x1, x2 = zipdata[0][0], zipdata[0][1]
        bs, c, w, h = x1.shape
        if(c>1):
          x1 = x1.reshape(bs,c,-1).float()
          x2 = x2.reshape(bs,c,-1).float()
        else:
          x1 =x1.reshape(bs, -1).float()
          x2 =x2.reshape(bs, -1).float()
        if args.embedding:
          x1=model_ae.encoder(x1.to(device))
          x2=model_ae.encoder(x2.to(device))
        x1=x1.double()
        x2=x2.double()
        #ortostep
        model.eval()
        if c>1:
          x2 = ChannelFussion(x2)
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
        bs_sample = x1.detach().to("cpu").numpy()
        W = to_graph(bs_sample,"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        W = torch.from_numpy(W).to(device).double()
        train_loss = criterion(Y,W)
        ACC, NMI, ARI, P = evalBasicRes( Y, zipdata[1].numpy(), args.n_clusters)
        trainData.setdata(data=[ACC,NMI,ARI,P])
        train_loss.backward()
        #更新梯度
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
          x =x.reshape( bs, -1) 
          x =  model_ae.encoder(x.float()).double()
      Y = model(x)
      if args.siam_metric:
        x = model_siam(x)
      bs_sample = x.detach().to("cpu").numpy()
      W = to_graph(bs_sample,"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
      W = torch.from_numpy(W).to(device).double()
      val_loss= criterion(Y,W)
      val_losst += val_loss.item()
      Y=Y.reshape(bs,-1)
      ACC, NMI, ARI, P = evalBasicRes( Y, tar.numpy(), args.n_clusters)
      datacont.setdata(data=[ACC,NMI,ARI,P])

    loss_t = round(loss_t / train_length,4)
    val_losst = round(val_losst/ test_length,4)
    valACC ,valNMI, valARI, valP = datacont.GetOneEpochResult(size=test_length)
    [*trainRows] = trainData.getdata(size=train_length)

    newrow=[' ',epoch, trainRows[0], valACC]
    csvRows.append(newrow)
    wind_loss.line([[loss_t,val_losst]],[epoch],win = 'old_train_loss',update = 'append')
    wind_acc.line([[trainRows[0],valACC]],[epoch],win = 'old_ACC',update = 'append')
    time.sleep(0.5)
    # 记录结果部分
    # 学习率终止策略
    scheduler.step(val_loss)
    act_lr = optimizer.param_groups[0]['lr']
    logging.info(f"epoch : {epoch + 1}/{args.epochs},trainACC={trainRows[0]},ACC={valACC},NMI={valNMI}, ARI={valARI},Purity={valP}  ,learning_rate ={act_lr}, train_loss = {round(loss_t,4)}")
    # if act_lr <= 1e-7:
    #   break
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
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows