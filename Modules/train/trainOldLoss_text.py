
import logging
import torch
import torch.optim as optim
from tqdm import tqdm 
from Modules.evaluation import evalBasicRes, evalBasicRes_text, evaluationForText
from Modules.loss import SpectralNetLoss
from Modules.model.attentionNet import AttentionNet3
from Modules.model.spectralNet import SpectralNet, SpectralNetText
from utils import exportExcel
from utils.basic import ChannelFussion, to_graph
from utils.data_contrainer import dataContainer, trainDataContainer

def train_SpectralNet_oldLoss_text( train_dataloader, test_dataloader, model_ae, args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = SpectralNetText(input_size=args.code_size, output_size=args.n_clusters).to(device)
  logging.info("================MODEL INFORMATION==============")
  logging.info(model)
  model_type=args.model_type+"_"+args.data
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = SpectralNetLoss()
  train_length = len(train_dataloader)
  datacont = dataContainer(epoch=args.epochs, len=4*3)
  epochTimes=0
  trainData = trainDataContainer(epoch=args.epochs, len=4*3) 
  csvRows = []
  csvRows.append(["SpctraNet_oldloss_text",'epoch', 'train_ACC', 'eval_ACC'])
  for epoch in range(args.epochs):
    loss_t = 0
    epochTimes+=1
    newrow=[]
    for zipdata in tqdm(train_dataloader, desc=f'train_Epoch [{epoch+1}/{args.epochs}]',total=train_length) :
        x1, x2 = zipdata[0], zipdata[1]
        bs, c = x1.shape
        x1 = x1.reshape(bs,-1).float()
        x2 = x2.reshape(bs,-1).float()

        if args.embedding:
          x1=model_ae.encoder(x1.to(device))
          x2=model_ae.encoder(x2.to(device))
        x1=x1.double()
        x2=x2.double()
        #ortostep
        model.eval()
        _ = model(x2,ortho_step=True)
        #gradstep
        model.train()
        optimizer.zero_grad()
        Y = model(x1)
        bs_sample = x1.detach().to("cpu").numpy()
        W = to_graph(bs_sample,"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
        W = torch.from_numpy(W).to(device).double()
        train_loss = criterion(Y,W)
        ACC, NMI, ARI, P = evalBasicRes( Y, zipdata[2].numpy(), args.n_clusters)
        trainData.setdata(data=[ACC,NMI,ARI,P])
        train_loss.backward()
        optimizer.step()
        loss_t += train_loss.item()     
    # scheduler.step(loss_t)
    act_lr = optimizer.param_groups[0]['lr']
    # 模型评估部分。
    model.eval()
    val_losst = 0
    test_length = len(test_dataloader)
    for  x, tar in test_dataloader:
      bs, c= x.shape
      x= x.to(device)
      #embedding
      if args.embedding:
          x =x.reshape( bs, -1) # 对注意力增加通道
          x =  model_ae.encoder(x.float()).double()
      Y = model(x)
      W = to_graph(x.reshape(bs,-1).detach().to("cpu").numpy(),"mean",None,args.n_neighbors,'k-hNNG',args.aprox).todense()
      W = torch.from_numpy(W).to(device).double()
      val_loss= criterion(Y,W)
      val_losst += val_loss.item()
      Y=Y.reshape(bs,-1)
      ACC, NMI, ARI, P = evalBasicRes_text( Y.detach().cpu().numpy(), tar.numpy(), args.n_clusters)
      datacont.setdata(data=[ACC,NMI,ARI,P])
    
    # 记录结果部分
    test_size = len(test_dataloader)
    test_ACC_t ,test_NMI_t, test_ARI_t ,test_Purity_t = datacont.GetOneEpochResult(size=test_size)
    logging.info(f"epoch : {epoch + 1}/{args.epochs},ACC={test_ACC_t},NMI={test_NMI_t},\
        ARI={test_ARI_t},Purity={test_Purity_t}  ,learning_rate ={act_lr}, train_loss = {round(loss_t,4)}")
    # 结果处理 部分
    loss_t = loss_t / train_length
    val_losst = val_losst/ test_length
    valACC ,valNMI, valARI, valP = datacont.GetOneEpochResult(size=test_length)
    [*trainRows] = trainData.getdata(size=train_length)
    newrow=[' ',epoch, trainRows[0], valACC]
    csvRows.append(newrow)
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
  exportExcel.export_csv(path=args.traindir+args.model_type+".csv", rowList=csvRows)
  # 保存模型
  torch.save(model, args.pretrainDir+model_type+".pkl")
  return rows