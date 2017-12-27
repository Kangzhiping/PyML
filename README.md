# Python Machine Learn 学习笔记
## 机器学习算法：
    
    估计器（estimator）：用于分类，聚类，和回归分析
    转换器（Transformer）: 用于数据预处理和数据转换
    流水线（Pipeline）: 组合数据挖掘流程，便于再次使用
    
    单模型方法：
    OneRule 方法：单一法则分类
	回归模型（logistic, 岭回归， lasso 回归）：找到一个分类的直线或者曲线	
	LR: 逻辑回归 logistic regression适合需要得到一个分类概率的场景，BGD(批量梯度下降算法)，SGD(随机梯度下降算法)，
	朴素贝叶斯：有某属性属于某类的概率= 某类有某属性的概率* 属于某类的概率/ 具有某属性的概率
		朴素贝叶斯的二值分类器：BernoulliNB
	K 近邻法：分类，用计算欧式距离，余弦夹角，曼哈顿距离，算法可以代入近邻个数，还有交叉检验方法
	决策树：以信息熵为度量构造一颗熵值下降最快的数，直到叶子节点处的熵值为0。选择信息增益最大(信息熵变化的大小)的属性为分类依据。信息增益： g(D,A) = 信息熵H(D) - 条件熵，A条件下D的信息熵 H(D|A)。 
		H(D)= - P(X) * LOG2 P(X), H(D|A) = P(Xi)-P(Xi|A) * log2 P(Xi|A).
	Apriori 关联规则挖掘算法： 亲和性分析，频繁二项集。还有Eclat（好用些）, FP-growth 亲和性分析算法。
	PCA: 主成分分析，返回方差从大到小排序。
	SVM: 支持向量机： 找到最佳分类的线
	神经网络: CNN, RNN
	
	多模型方法：
	1. Bagging
	RF: random forest 随机森林， 基于boostrap重抽样和随机选取特征,基预测器是决策树的集成方法
	   随机选取数据集和特征，形成多个决策树，取平均值，在大量数据集中占很大优势。能有效降低方差，得到的预测模型的总体正确率更高。GridSearchCV 类可以搜索最佳参数。
	2. Boosting(Adboost, Xgboost)
	3. Stacking
	
	SKlearn的一种检验方法评价分类效果： F1 值: p 准确率, q 召回率，调和平均数: 2*  p * q / p +q 
    轮廓系数（距离矩阵，不支持稀疏矩阵）： s = b -a / max(a,b)  a: 簇内距离， b: 簇间距离，与最近簇内各个个体之间的平均距离。接近最大值1：簇内相似度很高，接近0表示所有簇重合，-1表示分在错误的簇。

    距离度量：欧氏距离，闵可夫斯基距离，曼哈顿距离，切比雪夫距离，马哈拉 诺比斯距离
    相似度度量：向量空间余弦相似度(余弦夹角), 皮尔森相关系数,杰卡德相似系数，调整余弦相似度

    1. 监督学习： 分类预测，回归预测（用户点击购买预测，预测用户是否流失，预测房价 等等）
	    我们要学习一个模型，使得模型能够对任意给定的输入，对其相应的输出做出一个好的预测
    2. 无监督学习：聚类、相似度计算， 关联规则（新闻聚类，告警压缩，降维等等）
	    在学习时我们并不知道学习的结果是好是坏，仅仅是找出潜在的规律。
    3. 半监督学习：强化学习（无人机驾驶，AlphoGo 下棋策略等）
	    学习如何在环境中采取一系列行为，从而获得最大的累积回报。

## 深度学习：
    
    网络结构： lenet, googlenet, alexnet, lstm, GAN(生成对抗网络)
    使用Lasagne and nolearn 库来构建神经网络，数据处理由Theano完成。也可以用Pybrain, tensorflow, caffe 来构建。
    
    CNN: 卷积神经网络。图像处理： 人脸识别，无人驾驶。 受生物学上的感受野的机制而提出的一种前馈神经网络，擅长于在图片中寻找模式，并用他进行分类。
		最牛逼的地方在于通过感受野和权值共享减少了神经网络需要训练的参数个数。
		LeNet 网络结构： 卷积层，Pooling层(取最大值)，修正线性单元（激活函数 relu: rectified linear units：去掉负值f(x)=max(0,x)），全连接层
	RNN: 循环神经网络。序列问题的处理： 自然语言处理，语言识别，股票预测，文本生成。
    
    在线学习： 估计器的partial_fit 训练方法。
    
## 强化学习
    
    e.g. 学习玩游戏，逐渐学会怎么玩游戏
    
## 自然语言处理

    NLTK: Natural Language ToolKit, 处理文本词频。 自然语言处理。（N元语法，句法特征）
    CountVectorizer: 词袋模型向量方法。将文档转化为向量。
    TfIDFVectorizer: Term Frequency -  Inverse Document Frequency的缩写，即“词频-逆文本频率”，词频向量化预处理。
    Hash Trick: HashingVectorizer: 使用散列算法极大地降低了计算词袋模型所需要的内存开销。
    
## 图计算

    社交网络图（聚类分析）： 用杰卡德（A&B/A|B）相似系数寻找用户间的相似度,再用联通分支（NetworkX 内置的聚类方法，input 为阈值）
        把用户分成不同的簇，使用轮廓系数评价聚类效果（簇内距离最小和簇间距离最大）。

## 数据挖掘实施流程:
    
 1. 数据采集
 2. 数据处理
 3. 特征工程
 4. 模型选择
 5. 模型验证
 6. 模型优化
    