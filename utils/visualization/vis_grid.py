# 导入必要的包或方法，包括绘图的 Matplotlib 和生成数据的 make_blobs。
import matplotlib.pyplot as plt
# 导入生成数据的方法
import sklearn 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# 生成合成数据# n_samples是待生成的样本总数
# centers 表示要生成的样本中心（类别）数，或是确定的中心点数量

# 生成-网格图像

def visGrid(X_blobs,n_clusters=10, dpi=600, fileName="./output/imgs/vis_grid.jpg"):
    # (1)导入KMeans工具包
    # (2)构建 KMeans 模型对象，设定 k=4
    #当数据是tensor的时候才这样
    # X_blobs=X_blobs.detach().cpu().numpy()
    # bSize=X_blobs.shape[0]
    # p = np.random.permutation(bSize)
    # X_blobs = X_blobs[p]
    # #只用十分之一的数据
    # X_blobs=X_blobs[:int( bSize*0.1)]
    
    X_blobs=PCA(n_components=2).fit_transform(X_blobs)
    kmeans = KMeans(n_clusters=n_clusters)    
    kmeans.fit(X_blobs)
    # (4)绘制可视化图
    x_min, x_max = X_blobs[:, 0].min() - 1, X_blobs[:, 0].max() + 1
    y_min, y_max = X_blobs[:, 1].min() - 1, X_blobs[:, 1].max() + 1
    # (5)生成网格点矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='hermite', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired,aspect='auto', origin='lower')
    plt.plot(X_blobs[:, 0], X_blobs[:, 1], 'w.', markersize=5)
    # 用红色的x表示簇中心
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", cmap=plt.cm.Paired, s=5, linewidths=2,color='r', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks()
    plt.yticks()
    plt.show()
    plt.savefig(fileName, dpi=600 )
    print("===out put the ")
