# FHGNN: 模糊超图神经网络模型

## 一、项目简介
本项目基于论文《Fuzzy Representation Learning on Graph》与《Hypergraph Neural Networks》，提出一种融合超图神经网络（HGNN）与模糊逻辑的图数据表示学习模型，用于解决传统图模型在处理高阶相关性和特征不确定性时的局限性。模型支持节点分类、链路预测等任务，尤其在低标签率、高稀疏性数据集上表现优异。

## 二、目录结构:
```
FHGNN/
├── data/                # 数据集目录
│   ├── cora1000/        # Cora子集（1000节点）
│   └── ...              # 其他数据集
├── models.py/           # 模型定义
├── utils.py/            # 工具函数
├── run_fg.py            # 主运行脚本
├── tune_fg.txt          # 调参脚本
└── README.md            # 文档
```


## 三、环境配置
### 1.依赖安装
```
pip install -r requirements.txt
```
关键依赖:

torch>=1.8.0：深度学习框架

dgl>=0.8.0：图神经网络库

numpy>=1.19.0：数值计算

scipy>=1.5.0：科学计算

### 2.数据集准备
数据集	类型	节点数	特征维度	标签率	说明

cora1000	共引网络	1,000	1,433	10%	机器学习论文子集


citeseer	共引网络	1,498	3,703	15%	计算机科学论文

pubmed	医学共引	3,840	500	2%	低标签率典型数据集

### 3.执行命令与参数说明
主命令
```
python run_fg.py --dataset cora1000 --gpu_id 0
```
关键参数列表

参数名	类型	默认值	说明

--dataset	str	cora1000	数据集名称（需与data/目录下的子文件夹名一致）

--gpu_id	int	0	GPU 设备号（-1 表示使用 CPU）

--epochs	int	200	训练轮数

--lr	float	1e-3	学习率

--hidden_dim	int	256	隐藏层维度

--fuzzy_layers	int	2	模糊卷积层数（建议≤3，避免过平滑）

--k_neighbors	int	10	超边构建的近邻数（控制超图密度）

### 4.控制台输出
训练过程中会实时打印：
```
Epoch 100/200 | Loss: 0.352 | Acc: 82.5% (Train) | Val Acc: 78.9% | Time: 12.5s
```
测试结果保存至results/results.csv

## 四、复现此模型结果
步骤 1：下载数据集

步骤 2：启动训练
```
python run_fg.py --dataset cora1000 --gpu_id 0
```

## 五、引用
若您在研究中使用本模型，请引用以下论文：
```
@article{zhang2023fuzzy,
  title={Fuzzy Representation Learning on Graph},
  author={Zhang, Chun-Yang and Lin, Yue-Na and Chen, C. L. Philip and others},
  journal={IEEE Transactions on Fuzzy Systems},
  year={2023}
}

@inproceedings{feng2019hypergraph,
  title={Hypergraph Neural Networks},
  author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and others},
  booktitle={AAAI},
  year={2019}
}
```
