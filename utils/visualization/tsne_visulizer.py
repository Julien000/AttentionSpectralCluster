
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import (TSNE)

# 目前采用了两种配色方案，但是由于缺少一些背景知识，配色并不是很好看，color2的配色差异性较大，不容易有歧义
# 配色方案1
colors=[
    (1.0, 0.0, 1.0, 1.0),
    (0.5, 0.5, 0.0, 1.0),
    (0.0, 0.5, 0.5, 1.0),
    (0.5, 0.0, 0.5, 1.0),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 1.0, 0.5, 1.0),
    (0.5, 1.0, 0.5, 1.0),
    (0.0, 0.0, 1.0, 1.0),
    (0.5, 0.5, 0.5, 1.0),
    (1.0, 0.4, 0.4, 1.0)
]
# 配色方案
colors2=[
    (255/255.0, 0, 0),
    (255/255.0, 0, 255/255.0),
    (128/255.0, 128/255.0, 128/255.0),
    (64/255.0, 0, 255/255.0),
    (0, 255/255.0, 255/255.0),
    (0, 255/255.0, 0),
    (191/255.0, 255/255.0, 0),
    (255/255.0, 255/255.0, 0),
    (255/255.0, 191/255.0, 0),
    (255/255.0, 128/255.0, 0)
    # (255/255.0, 0, 191/255.0),
]
def plot_embedding(X,y,k ,title=None):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)
    for digit in range(k):# 因为现在只有mnist 数据集只用10个标记
        # marker=f"${digit}$",
        ax.scatter(
            *X[y == digit].T,
            s=5, #标记大小
            color=colors2[digit], # 标记颜色
            alpha=0.8, #透明度0.425
            zorder=2,
        )
    if title!=None:
        ax.set_title(title,loc="center")
    ax.axis("off")



def VisTSNE(X, y, outdir="", outName="rowdata.jpg",test_size=10000, k=10, isTitle=False, dpi=300):
    '''
        X:  [n_smaple , n_features] 样本列
        y: [n_sample] ground truth
        outdir: 输出的目录 output/image/
        outName: 输出的文件名
        test_size: 样本的数量。 针对mnist-测试数据集可忽略
        k: 具体分类的类别个数， n_classes
        dpi: 输出的图片质量，越高质量越好。
        因为学术论文通常需要的都是，svg图像本身是可以自己设置多类型图像，但是懒了，需要commit任务才能去作
    '''
    #TODO: 对于title 位置的控制
    #TODO: 加速计算部分没有完成
    X =  X.reshape(test_size,-1)
    embeddings = {
        "t-SNE embeedding": TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            n_iter=500,
            n_iter_without_progress=150,
            n_jobs=2,
            random_state=1,
        )
    }
    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        if name.startswith("Linear Discriminant Analysis"):
            data = X.copy()
            data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
        else:
            data = X
        print(f"Computing {name}...")
        start_time = time()
        projections[name] = transformer.fit_transform(data, y)
        timing[name] = time() - start_time

    for name in timing:
        if isTitle==True:
            title = outName
            plot_embedding(projections[name], y, k, title=title)
        else:
            plot_embedding(projections[name], y, k)


    # plt.show()
    plt.savefig(outdir+outName, format="svg", dpi=dpi )
    
def VisTSNE_test(X, y, outdir="", outName="rowdata.jpg",test_size=10000, k=10,dpi=300):
    '''
        X:  [n_smaple , n_features] 样本列
        y: [n_sample] ground truth
        outdir: 输出的目录 output/image/
        outName: 输出的文件名
        test_size: 样本的数量。 针对mnist-测试数据集可忽略
        k: 具体分类的类别个数， n_classes
        dpi: 输出的图片质量，越高质量越好。
        维护版本代码，试图采用加速的计算模型。
    '''
    #TODO: 对于title 位置的控制
    #TODO: 加速计算部分没有完成
    #TODO: 对于展示的细节效果，如标点的大小。
    X =  X.reshape(test_size,-1)
    embeddings = {
        "t-SNE embeedding": TSNE(
            n_jobs=2,
            random_state=0
        )
    }
    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        if name.startswith("Linear Discriminant Analysis"):
            data = X.copy()
            data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
        else:
            data = X
        print(f"Computing {name}...")
        start_time = time()
        # projections[name] = transformer.fit(data, y)
        projections[name] = transformer.fit(data)
        timing[name] = time() - start_time

    for name in timing:
        title = outName
        plot_embedding(projections[name], y, k, title)
    plt.show()
    plt.savefig(outdir+outName, dpi=dpi )
