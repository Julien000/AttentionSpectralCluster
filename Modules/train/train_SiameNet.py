import torch
from torch import nn
import torch.optim as optim
from Modules.loss import ContrastiveLoss
from Modules.model.siameNet import SiameseNet
from utils.pairs import create_pairs_from_unlabeled_data
from sklearn.model_selection import train_test_split
class Dataset_Siemes(object):
    #numpy -> tensor
    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return (self.x0[index],
                self.x1[index],
                self.label[index])

    def __len__(self):
        return self.size

# 专门设计给孪生网络使用的东西。
def siamesedataloader(X, n_neighbors, test_size=0.2,batch_size=128,use_approx=False):
  '''
  input:
  X - numpy.ndarray
  
  n_neighbors: int
      -number of neighbors
  test_size: [0,1] int, default=0.2
      - size of test set
  batch_size: int, default=128
  
  return:
  train_dataloader,val_dataloader - torch dataloader
  '''
  #assert type(X) == np.ndarray, "X is not np.ndarray"
  
  pairs = create_pairs_from_unlabeled_data(X,k=n_neighbors,use_approx=use_approx)

  X_train, X_val, y_train, y_val = train_test_split(pairs[0], pairs[1], test_size=test_size)

  X1_train,X2_train,y_train = X_train[:,0,:],X_train[:,1,:],y_train
  X1_val,X2_val,y_val = X_val[:,0,:],X_val[:,1,:],y_val

  train_data = Dataset_Siemes(X1_train, X2_train, y_train)
  train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True)

  val_data = Dataset_Siemes(X1_val, X2_val, y_val)
  val_dataloader = torch.utils.data.DataLoader(val_data,batch_size=batch_size, shuffle=True)
  return train_dataloader, val_dataloader

# 训练SiameseNet
def train_SiameseNet(input_size,output_size,train_dataloader,val_dataloader,file = "output/pretrain/SIAME/Mnist.ckp",epochs = 200,verbose=False):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = SiameseNet(input_size=input_size,output_size=output_size).to(device)
  optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
  criterion = ContrastiveLoss(1)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

  for epoch in range(epochs):
      loss_t = 0
      model.train()
      for x1,x2,labels in train_dataloader:
          x1,x2,labels = x1.to(device),x2.to(device),labels.to(device)
          
          optimizer.zero_grad()
          
          z1, z2  = model(x1,x2)
          train_loss = criterion(z1, z2, labels)

          train_loss.backward()
          
          optimizer.step()
          
          loss_t += train_loss.item()

      loss_t = loss_t / len(train_dataloader)
      loss_v = 0
      model.eval()
      for x1,x2,labels in val_dataloader:
          x1,x2,labels = x1.to(device),x2.to(device),labels.to(device)
          
          z1, z2 = model(x1,x2)
          val_loss = criterion(z1, z2, labels)
          
          loss_v += val_loss.item()

      
      loss_v = loss_v / len(val_dataloader)
      scheduler.step(val_loss)
      act_lr = optimizer.param_groups[0]['lr']
      if verbose:
        print("epoch : {}/{}, learning_rate {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, epochs,act_lr, loss_t, loss_v))
      if act_lr <= 1e-7:
        break
  if file!= None:
    torch.save(model, file)
  return model