```

```

# 1、目录结构

```
Linux 
|-- assets # 静态资源目录
|   |-- config # 执行参数目录
|   |-- datasets # 数据集目录
|   `-- env #虚拟环境目录
|-- Modules #模型目录
|   |-- model # 模型目录
|   |   |-- attentionNet.py
|   |   |-- autoencoder.py
|   |   |-- selfAttention.py
|   |   `-- spectralNet.py
|   `-- train # 训练方法目录
|       |-- trainL21Loss.py
|       |-- trainOldLoss.py
|       `-- train_AE.py
|   |-- evaluation.py
|   `-- loss.py
|-- output  # 输出目录，主要包含预训练模型， 日志文件和 数据表格
|   |-- excel #批量化结果
|   |-- logInfo # 日志
|   `-- pretrain # 预训练模型
|-- utils # 通用方法目录
|-- compare_model #对比模型方法
|-- README.md
|-- inference.py
|-- main_image.py # 针对的是 图片数据集启动文件
|-- main_text.py # 针对的是 文本是数据集入口脚本
`-- requirements.txt
```

# 2、配置环境

```
pyhton =3.7
pytorch = 1.11 
other 环境 可查看 assets/requirement.txt 并通过指令安装 ，pip install -r  assets/requirement.txt
```

# 2、 Train 训练模型

现在对mian.py 文件进行修改 **main_image.py** 代表的是图片数据集的入口，**mian_text.py**代表的是文本数据集的入口。

## 调试运行程序方式

1. 执行训练程序从 `main.py`开始执行
2. 选择需要的数据集可以在 `/config` 目录中的文件配置 `data`参数，选择需要执行的数据集

## 命令行运行程序方式

1. 进入到正确的虚拟环境中
2. 输入如下指令：即可执行程序

```
python main.py --config batch_config.yaml  
```

# 3、评估模型部分 TODO

1. 执行训练程序从 `inference.py`开始执行
2. 选择需要的数据集可以在 `/config` 目录中的文件配置 `data`参数，选择需要执行评估的数据集，然后将对应的模型加载进去评估
