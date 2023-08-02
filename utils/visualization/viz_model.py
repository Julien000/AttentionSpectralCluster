from torchviz import make_dot
from Modules.model.attentionNet import AttentionNet
import torch
import torchvision.models as models
from torchinfo import summary
#
def vizModel(x,y,model,dir= "./output/imgs",filetype='jpg'):
    '''
    x 一组数据
    y 对应x 经过模型输出的结果
    model 模型
    dir= "./output/imgs" #图片保存路径
    filetype='jpg' # 图片类型。
    注意; 这个包需要在操作系统上安装 graphviz 程序
    然后再通过 pip 
    # 针对可视化需要安装Successfully installed graphviz-0.20.1 torchviz-0.0.2
    # 并且需要再服务器安装新的 graphviz 这个可视化环境，属于操作系统的
    '''
    # x = torch.randn(1, 1, 784).double().requires_grad_(True).cuda()# 定义一个网络的输入值
    # model = AttentionNet(input_size=784, output_size=2).cuda()
    # _ = model(x,ortho_step=True )    # 获取网络的预测值
    # y = model(x)    # 获取网络的预测值
    MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    MyConvNetVis.format = filetype
    # 指定文件生成的文件夹
    MyConvNetVis.directory =dir
    # 生成文件
    MyConvNetVis.view()

    print("----output model vizlization -----")


def vizModelInfo(model,inputSize):
    '''
    pip install torchinfo ----
    需要安装对应的可视化package
    -目前还没有控制好，到时候，需要将一些参数传入。
    输入模型
    然后，输入模型需要输入的数据维度，得到模型的架构图。
    '''
    # resnet18 = models.resnet18() # 实例化模型
    # x = torch.randn(1, 1, 784).double().requires_grad_(True).cuda()# 定义一个网络的输入值
    # model = AttentionNet(input_size=784, output_size=10).cuda()
    # _ = model(x,ortho_step=True )    # 获取网络的预测值
    # y = model(x)    # 获取网络的预测值
    # model = (input_size=784, output_size=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # attNetInfo = summary(model, x,device=device,batch_dim=10, dtypes=[torch.float64]) # 1：batch_size 3:图片的通道数 224: 图片
    attNetInfo = summary(model,input_size=inputSize,device=device, dtypes=[torch.float64]) # 1：batch_size 3:图片的通道数 224: 图片
    
    return attNetInfo
    # print(attNetInfo)
    # print("=============log info success!!!!")

