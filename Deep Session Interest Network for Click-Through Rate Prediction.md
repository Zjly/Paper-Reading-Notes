# Deep Session Interest Network for Click-Through Rate Prediction

### 基于深度会话兴趣网络的点击率预测

###### Published in: International Joint Conferences on Artificial Intelligence Organization

###### Paper address: https://www.ijcai.org/proceedings/2019/319

### 概述

从用户的行为序列中捕捉用户动态的、不断发展的兴趣是点击率（CTR）预测的重要的研究方向之一。用户的行为序列按时间可以被分隔为一个个的会话，在不同的会话中用户的行为有着相异性，而在同一个会话中用户的行为是高度同质的。

论文基于用户行为序列的特点，提出了一种新的CTR模型：深度会话兴趣网络（DSIN）。DSIN将用户的连续行为划分为多个会话，使用带有偏差编码的自注意力网络捕获会话行为的内在交互性和相关性，并提取每个会话中的用户兴趣。之后，DSIN使用双向长短期记忆神经网络（Bi-LSTM）建模用户兴趣在不断变化的不同会话之间的交互与进化。由于不同的会话兴趣对于目标项有着不同的影响，所以使用局部激活单元自适应的让会话兴趣与目标项进行聚合，形成行为序列的最终表示。

论文主要贡献：

- 强调了用户行为在不同会话中的相异性和在同一会话中的同质性，并以此提出了DSIN；
- 设计了带有偏差编码的自注意力网络获取每个会话中的准确兴趣表达，使用Bi-LSTM捕获历史会话的顺序关系，采用局部激活单元聚合不同会话兴趣对目标项的影响；
- 经过比较试验发现，与现有的CTR模型相比，DSIN有着一定的优越性。

### 深度会话兴趣网络

##### 基础模型

###### 特征表示

- 用户信息
  - 性别
  - 城市
  - ……
- 物品信息
  - 卖家id
  - 品牌id
  - ……
- 用户行为
  - 用户最近点击物品的物品id

###### embedding层

- embedding层：$\mathbf E \in \mathbb R^{M \times d_{model}}$，$M$代表稀疏特征的大小
- 用户信息：$\mathbf X^U \in \mathbb{R}^{N_u \times d_{model}}$，$N_u$代表用户稀疏特征的大小
- 物品信息：$\mathbf X^I \in \mathbb{R}^{N_i \times d_{model}}$​，$N_i$​​代表物品稀疏特征的大小
- 用户行为：$\mathbf S=[\mathbf{b}_1;...;\mathbf{b}_i;...;\mathbf{b}_N] \in \mathbb R^{N \times d_{model}}$，$N$代表用户历史行为次数，$\mathbf b_i$代表第$i$次行为的embedding​

###### 多层感知机

- 从用户信息、物品信息和用户行为中嵌入稀疏特征
- 通过激活函数将embedding特征连接、flattened并将它们输入到多层感知机中
- 使用softmax函数预测用户点击目标物品的概率

###### 损失函数

使用负对数似然函数作为损失函数，其数学表示形式如公式1所示：
$$
L=-\frac{1}{N} \sum\limits_{(x,y) \in \mathbb{D}} (y\log{p(x)}  + (1-y)\log{(1-p(x))}) \tag{1}
$$
其中$\mathbb{D}$是训练数据集，网络输入$[\mathbf X_u,\mathbf X^I,\mathbf S]$被表示为$x$，$y \in \{0,1\}$表示用户是否点击了商品，$p(\cdot)$表示用户点击商品的预测概率。

##### DSIN

![image-20210730221836298](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301539767.png)

如图所示，DSIN模型由以下三部分组成：

- 用户信息模型
- 物品信息模型
- 用户行为模型
  - 会话划分层：将用户行为序列划分为会话
  - 会话兴趣提取层：提取用户的会话兴趣
  - 会话兴趣交互层：捕获会话兴趣之间的顺序关系
  - 会话兴趣激活层：将本地交互单元应用于用户的会话兴趣的目标项上

###### 会话划分层

DSIN会将用户行为序列$\mathbf S$划分为会话$\mathbf Q$，其中第k次会话$\mathbf Q_k=[\mathbf{b}_1;...;\mathbf{b}_i;...;\mathbf{b}_T]\in \mathbb R^{T \times d_{model}}$，T是会话内保留的行为数，$\mathbf b_i$是用户在会话中的第i次行为。用户会话分割存在于间隔时间大于30分钟的相邻行为之间，也就是说大于30分钟的相邻行为会被分在两个不同的用户会话之中，这两个会话就是分割点。

###### 会话兴趣提取层

同一个会话中，用户的行为应该是高度相关的，但是会话中有可能存在着用户的随意行为，这些随意行为会影响到会话的正确表达。所以，为了更准确的捕获会话中行为的关系，DSIN使用了多头自注意力机制来实现目标。

###### 偏差编码

为了捕获会话的顺序关系以及不同子空间中存在的偏差，论文提出了位置编码的基础偏差编码$\mathbf{BE}\in \mathbb R^{K \times T \times d_{models}}$​​，$\mathbf{BE}$​​中的每个元素定义如公式2所示：
$$
\mathbf{BE}_{(k,t,c)}=\mathbf{w}^K_k+\mathbf{w}^T_t+\mathbf{w}^C_c \tag{2}
$$
其中，$\mathbf{w}^K \in \mathbb R^K$​​是会话的偏差向量，$\mathbf{w}^T \in \mathbb R^T$​​是位置的偏差向量，$\mathbf{w}^C \in \mathbb R^{d_{model}}$​​​​​是用户行为embedding中单元位置的偏差向量。用户的行为会话$\mathbf Q$​更新如公式3所示：
$$
\mathbf Q = \mathbf Q + \mathbf{BE} \tag{3}
$$

###### 多头注意力机制

用户行为$\mathbf Q_k=[\mathbf Q_{k1};...;\mathbf Q_{kh};...;\mathbf Q_{kH}]$​，其中$\mathbf Q_{kh} \in \mathbb{R}^{T \times d_h}$是$\mathbf Q_k$​的第h个自注意力头，H是自注意力头的数量，$d_h=\frac{1}{h}d_{model}$​。第h个自注意力头的输出如公式4所示：
$$
\begin{aligned}
\mathbf{head}_h&=Attention(\mathbf{Q}_{kh}\mathbf{W}^Q,\mathbf{Q}_{kh}\mathbf{W}^K,\mathbf{Q}_{kh}\mathbf{W}^V) \\
&=softmax(\frac{\mathbf{Q}_{kh}\mathbf{W}^Q\mathbf{W}^{K^{T}}\mathbf{Q}_{kh}^{T}}{\sqrt{d_{model}}})\mathbf{Q}_{kh}\mathbf W^V
\end{aligned}
\tag{4}
$$
然后，将各个自注意力头的矢量连接起来并输入前馈网络，其数学表示如公式5所示：
$$
\mathbf I^Q_k=FFN(Concat(\mathbf{head}_1,...,\mathbf{head}_H)\mathbf W^O) \tag{5}
$$
其中FFN是前馈神经网络，$\mathbf W^O$​是线性矩阵。用户第k次会话的兴趣$\mathbf I_k$​的计算如公式6所示：
$$
\mathbf I_k=AVG(\mathbf I^Q_K) \tag{6}
$$

###### 会话兴趣交互层

Bi-LSTM可以捕获到顺序关系，并可以应用于DSIN中的会话兴趣交互建模，基础LSTM的公式表达形式在这里就不在赘述。而双向LSTM之中存在着正向的RNN和反向的RNN，隐藏状态$\mathbf H$的计算形式如公式7所示，其中$\overrightarrow{\mathbf {h}_{ft}}$是正向LSTM的隐藏状态，$\overleftarrow{\mathbf {h}_{bt}}$是反向LSTM的隐藏状态：
$$
\mathbf H_t=\overrightarrow{\mathbf {h}_{ft}} \oplus \overleftarrow{\mathbf {h}_{bt}} \tag{7}
$$

###### 会话兴趣激活层

用户的会话兴趣与目标物品的关联程度越高，对用户是否点击目标物品的影响就越大。用户会话兴趣的权重需要根据目标物品重新分配，所以DSIN使用了注意力机制来分配权重，其数学表示如公式8所示：
$$
\begin{split}
& a^I_k=\frac{exp(\mathbf{I}_k\mathbf{W}^I\mathbf{X}^I)}{\sum^K_kexp(\mathbf{I}_k\mathbf{W}^I\mathbf{X}^I)} \\
& \mathbf{U}^I = \sum\limits^K_k a^I_K \mathbf I_K
\end{split}
\tag{8}
$$
同样，物品权重计算如公式9所示：
$$
\begin{split}
& a^I_k=\frac{exp(\mathbf{H}_k\mathbf{W}^H\mathbf{X}^I)}{\sum^K_kexp(\mathbf{H}_k\mathbf{W}^H\mathbf{X}^I)} \\
& \mathbf{U}^H = \sum\limits^K_k a^H_K \mathbf H_K
\end{split}
\tag{9}
$$
最后，将用户信息和物品信息的embedding和$\mathbf{U}^I$和$\mathbf{U}^H$连接输入到MLP层中。

##### 总结

论文指出了用户的连续行为会由多个历史会话组成，并且用户行为在同会话中同构，在跨会话中异构。在此基础上，论文提出了深度会话兴趣网络(DSIN)：首先利用带有偏差编码的自注意力机制提取用户对每个会话的兴趣，然后使用Bi-LSTM来捕获上下文会话兴趣的顺序关系，最后利用局部激活单元聚合用户对目标项的不同会话兴趣表示。

