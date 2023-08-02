import torch


# 保存断点代码
# 该算法可能不适用于多GPU

#TODO
'''
1、保存断点功能 DONE
2、加载断点功能 DONE
3、 GPU、CPU、 方案： 通过DEVICE动态检测环境LoadModel
5、多GPU适配--目前代码不支持
'''

def SaveCKP(model, PATH, sType="DICT", saveMap={}):
    '''
    参数解释
        必备参数：
            model： 被存储的模型
            PATH： 保存的断点位置 
        可选参数
            sType： 选择保存的断点类型。
            1、DICT ：保存完整模型的权重
            2、ALL： 直接保存完整的模型
            3、"other": 通过自定义的映射保存信息如epoch等。
            {
                epoch: n,
                lr:0.001
                ...
                #自定义的信息
            }
    '''
    if sType == 'DICT':#保存state_dict 推荐
        torch.save(model.stat_dict(), PATH)
    elif sType == 'ALL': #保存模型
        torch.save(model, PATH)
    else:
        saveMap['model_state_dict']=model.stat_dict()
        torch.save(saveMap, PATH)


def LoadCKP(PATH, sType="DICT",model=None):
    '''
    参数解释
        必备参数：
            PATH： 加载的断点位置 
        可选参数
            sType： 选择保存的断点类型。
            1、DICT ：保存完整模型的权重-- 训练好的保存
            2、ALL： 直接保存完整的模型-- 训练好的保存
            3、"other": 通过自定义的映射保存信息如epoch等。 --用于训练中断加载
            {
                epoch: n,
                lr:0.001
                ...
                #自定义的信息
            }
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sType == 'DICT':#保存state_dict 推荐
        torch.load_state_dict(torch.load( PATH),map_location=device)
        return  
    elif sType == 'ALL': #保存模型
         return  torch.load(PATH, map_location=device).to(device)
    else:
        ckp = torch.load(PATH)
        star_epoch = ckp['epoch'] #断点停止的epoch
        assert(model == None) , "Missing required parameter 'model' "
        model.load_state_dict(ckp['model_state_dict'])

        return model, star_epoch
