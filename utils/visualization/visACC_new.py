import matplotlib.pyplot as plt
savepath="beta_exp.svg"
# savepath="beta_exp.png"
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
#!/usr/bin/env python3
import pandas as pd
import numpy as np
#读取工作簿和工作簿中的工作表
# writer_1=pd.ExcelFile('vizdata/USPS_Parameters__2023-04-26.xls')
# data_frame= writer_1.parse('Sheet1')
# ACC = list(data_frame['ACC'])
# NMI = list(data_frame['NMI'])
# ARI = list(data_frame['ARI'])
# p = list(data_frame['purity'])
# beta = list(data_frame['beta'])
# X =[0,1	,2	,3	,4, 5, 6, 7, 8 ,9]
# ACC	=[0.6813,	0.897,	0.91,	0.885,	0.894, 0.859]
# NMI =[0.775,	0.843,	0.853	,0.818	,0.816, 0.795]
# ARI	=[0.61	,0.811	,0.834	,0.784	,0.793, 0.755]
# ACC=[68.13 ,	89.70 	,91.00 	,88.50 	,89.40 	,85.90 ]
# NMI=[77.50 ,	84.30 	,85.30 	,81.80 	,81.60 	,79.50 ]
# ARI=[61.50 ,	81.10 	,83.40 	,78.40 	,79.30 	,75.50 ]


fig,ax=plt.subplots(2,3, figsize=(12, 8))
# fig=plt.figure(figsize=(12,6))

# plt.figure(figsize=(8,6))
# plt.subplot(111)
# plt.ylim((60,100))
# plt.ylabel('评价指标 %')
# plt.xlabel('(a) mnist')
code = ['(a) MNIST', '(b) QMNIST', '(c) Digits', '(d) Fashion', '(e) Letters', '(f) USPS']
xls_list = [
    "vizdata/Minist_Parameters__2023-04-26.xls",
    "vizdata/QMNIST_Parameters__2023-04-27.xls",
    "vizdata/Digits_Parameters__2023-04-27.xls",
    "vizdata/FashionMNIST_Parameters__2023-04-27.xls",
    "vizdata/Letters_Parameters__2023-04-29.xls",
    'vizdata/USPS_Parameters__2023-04-26.xls',
]
j=0
for i in range(6):
    writer_1=pd.ExcelFile(xls_list[i])
    data_frame= writer_1.parse('Sheet1')
    ACC = list( np.round(data_frame['ACC']*100,2))
    NMI = list( np.round(data_frame['NMI']*100,2))
    ARI = list( np.round(data_frame['ARI']*100,2))
    p = list( np.round(data_frame['purity']*100,2))
    # ACC = list(np.int32(np.round(data_frame['ACC']*100)))
    # NMI = list(np.int32(np.round(data_frame['NMI']*100)))
    # ARI = list(np.int32(np.round(data_frame['ARI']*100)))
   

    p = list( np.int32(np.round(data_frame['purity']*100)))
    beta = list(data_frame['beta'])
    if(i<3):
        ax[0][i].set_xlabel(code[i])
        ax[0][i].plot(beta, ACC, color="blue", linewidth=1, linestyle="-" ,marker='o')
        ax[0][i].plot(beta, NMI,  color="red", linewidth=1, linestyle="-",marker='s')
        ax[0][i].plot(beta, ARI, color="green", linewidth=1, linestyle="-",marker='^')
        
        ax[0][i].set_ylim(20, 98)
        ax[0][i].legend(['ACC','NMI', 'ARI'], loc='lower right')
    else:

        ax[1][j].set_xlabel(code[i])
        ax[1][j].plot(beta, ACC, color="blue", linewidth=1, linestyle="-" ,marker='o')
        ax[1][j].plot(beta, NMI,  color="red", linewidth=1, linestyle="-",marker='s')
        ax[1][j].plot(beta, ARI, color="green", linewidth=1, linestyle="-",marker='^')
        ax[1][j].set_ylim(10, 98)
        ax[1][j].legend(['ACC','NMI', 'ARI'], loc='lower right')
        j+=1

plt.savefig(savepath, format="svg", dpi=300 , bbox_inches='tight' )
# plt.show()
# plt.savefig(savepath, dpi=300 )
print(f"file {savepath} successful !!!")