import numpy as np
import matplotlib.pyplot as plt


# plt.figure(figsize=(6,5), dpi=600)
# plt.subplot(111)

# X = [0,0.5, 1, 2, 3, 4]
# ACC = [0.695, 0.841,0.895, 0.842,0.853,0.916]
# ARI = [0.609, 0.753, 0.795, 0.735, 0.754, 0.831]
# NMI = [0.77,0.793, 0.821, 0.779, 0.792, 0.839]


# plt.plot(X, ACC,'d-', color="blue", linewidth=2.5, linestyle="-" )
# plt.plot(X, NMI, 'd-' , color="red", linewidth=2.5, linestyle="-")
# plt.plot(X, ARI, 'd-' , color="green", linewidth=2.5, linestyle="-")
# # plt.plot(X, ACC,color="blue", linewidth=2.5, linestyle="-" )
# # plt.plot(X, NMI, color="red", linewidth=2.5, linestyle="-")
# # plt.plot(X, ARI,  color="yellow", linewidth=2.5, linestyle="-")

# # plt.title('Default color cycle')
# # plt.set_title('plt.cm.Paired colormap')


# ======
import numpy as np
import matplotlib.pyplot as plt
# 导入 csv 库
import csv
import xlrd
# 以读方式打开文件


def viz_fromxlsTrain(filePath,
    dpi=600,sheet_data="Sheet1"):
    '''
     sheet_data: meaning the file pages
     默认会将第一行去除掉作为表头-然后后面的东西作为自己的内容。
    '''
    ARI = []
    ACC = []
    NMI = []
    savepath=filePath.split('.',1)[0]+'.jpg'
    # 打开文件
    workbook=xlrd.open_workbook(filePath)
    # 读取sheet页
    sheet=workbook.sheet_by_name(sheet_data)
    # 获取表的行列数
    rows=sheet.nrows
    cols=sheet.ncols
    # 获取表中数值
    for row in range(1,rows):
        ACC.append( float(sheet.cell(row,3).value))
        NMI.append( float(sheet.cell(row,4).value))
        ARI.append( float(sheet.cell(row,5).value))
    plt.figure(figsize=(4,3), dpi=dpi)
    plt.subplot(111)
    plt.ylim((0,1))
    X = np.linspace(0, 200, num=200, endpoint=True)
    plt.plot(X, ACC, color="blue", linewidth=1.5, linestyle="-" )
    plt.plot(X, NMI,  color="red", linewidth=1.5, linestyle="-")
    plt.plot(X, ARI, color="green", linewidth=1.5, linestyle="-")
    # plt.plot(X, ACC,color="blue", linewidth=2.5, linestyle="-" )
    # plt.plot(X, NMI, color="red", linewidth=2.5, linestyle="-")
    # plt.plot(X, ARI,  color="yellow", linewidth=2.5, linestyle="-")

    # plt.title('Default color cycle')
    # plt.set_title('plt.cm.Paired colormap')
    plt.legend(['ACC','NMI', 'ARI'],loc='lower right')
    

    plt.show()
    plt.savefig(savepath, dpi=600 )
    print(f"file {savepath} successful !!!")

def viz_pltTrain(savepath="test.jpg", dpi=600):

    X = [0, 1, 2, 2.5, 3, 3.5, 4, 5]
    ACC = [0.719, 0.891, 0.954, 0.954, 0.891, 0.957, 0.953,0.954]
    NMI = [0.811, 0.849, 0.89, 0.891, 0.85, 0.896, 0.889, 0.891]
    ARI = [0.687, 0.823, 0.903, 0.903, 0.826, 0.908, 0.901, 0.903]

    plt.figure(figsize=(4,3), dpi=dpi)
    # plt.subplot(111)
    plt.ylim((0,1))
    # 空心标记
    # plt.plot(X, ACC,'d', color="blue", linewidth=1.5, linestyle="-" ,markerfacecolor='white')
    # plt.plot(X, NMI,'d',  color="red", linewidth=1.5, linestyle="-",markerfacecolor='white')
    # plt.plot(X, ARI,'d', color="green", linewidth=1.5, linestyle="-",markerfacecolor='white')
    plt.plot(X, ACC,'d', color="blue", linewidth=1.5, linestyle="-")
    plt.plot(X, NMI,'d',  color="red", linewidth=1.5, linestyle="-")
    plt.plot(X, ARI,'d', color="green", linewidth=1.5, linestyle="-")
    plt.legend(['ACC','NMI', 'ARI'],loc='lower right')

    plt.show()
    plt.savefig(savepath, dpi=600 )
    print(f"file {savepath} successful !!!")

#
def testvis_epoch():
    pathlist=[
    "exp_result/save_InferModel/beta_sanet/train_epoch/AttentionNet_qmnist_beta_AttentionN_L21QMNIST_beta0.xls",
    "exp_result/save_InferModel/beta_sanet/train_epoch/AttentionNet_qmnist_beta_AttentionN_L21QMNIST_beta1.xls",
    "exp_result/save_InferModel/beta_sanet/train_epoch/AttentionNet_qmnist_beta_AttentionN_L21QMNIST_beta2.xls",
    "exp_result/save_InferModel/beta_sanet/train_epoch/AttentionNet_qmnist_beta_AttentionN_L21QMNIST_beta25.xls",
    "exp_result/save_InferModel/beta_sanet/train_epoch/AttentionNet_qmnist_beta_AttentionN_L21QMNIST_beta3.xls",
    "exp_result/save_InferModel/beta_sanet/train_epoch/AttentionNet_qmnist_beta_AttentionN_L21QMNIST_beta35.xls",
    "exp_result/save_InferModel/beta_sanet/train_epoch/AttentionNet_qmnist_beta_AttentionN_L21QMNIST_beta4.xls",
    "exp_result/save_InferModel/beta_sanet/train_epoch/AttentionNet_qmnist_beta_AttentionN_L21QMNIST_beta5.xls"
]
    # for P in pathlist:
    #     viz_fromxlsTrain(filePath=P)
    viz_pltTrain()