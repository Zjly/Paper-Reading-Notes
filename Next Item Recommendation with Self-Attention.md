# Next Item Recommendation with Self-Attention

### 基于自注意力机制的下个项目推荐

### Introduction

利用用户的历史数据是进行用户未来行为预测的有效手段之一，循环神经网络RNN与卷积神经网络CNN等神经网络模型能够有效地建立模型并进行预测。但是，RNN与CNN无法明确的捕获整个用户历史记录中项目与项目间的交互信息，这些信息能够让模型理解单个项目对之间的细粒度关系。

为了解决上述问题，论文提出了一种神经序列推荐系统，能够在对连续的项目进行建模的同时，对当前窗口中所有用户的交互信息进行建模，其模型能够被视作一种”local-global“的方法，有效的捕获了RNN与CNN中难以捕获的全局性的交互信息。论文的模型基于自注意力机制与度量嵌入，在用户历史记录中显式的调用项目之间的交互信息，使得模型能够同时兼顾长程的全局信息与短程的k-consecutive信息。

### ATTREC Model

论文模型由两部分所组成：

- 模拟用户的短期意图的*self-attention module*
- 模拟用户长期偏好的*collaborative metric learning*组件

##### Sequential Recommendation

$\mathcal{U}$代表一组用户，$\mathcal{I}$表示一组项目，其中$|\mathcal{U}|=M$，$|\mathcal{I}|=N$。当$\mathcal{H}^u \subseteq \mathcal{I}$时，使用$\mathcal{H}^u=(\mathcal{H}^u_1,...,\mathcal{H}^u_{|\mathcal{H}^u|})$来表示按时间序列排列的、用户与项目交互的项目序列。序列推荐的目标是基于用户之前的与项目交互的轨迹，预测用户将与之交互的下一个项目。

##### Short-Term Intents Modelling with Self-Attention

用户在短时间内的交互信息有效的反映了用户近期的需求或是意图，因此，对用户短期交互信息建模是了解用户时间偏好的有效手段。自注意力机制在捕捉序列模式方面有着良好的效果，论文模型使用了自注意力机制为用户短期交互信息进行了建模，论文方法的自注意力机制模块如下图所示：

![image-20210626155358840](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301540981.png)

自注意力机制是注意力机制的一种特例，它通过单个序列与其自身进行匹配来进行表示，可以保留上下文信息并无视距离捕获元素之间的关系。注意力机制模块的输入由*query,key,value*所组成，其输出为*value*的加权和，其中权重由*query*和*key*所决定。在自注意力机制模型中，*query,key,value*是由用户最近的交互历史组成，并且它们是相同的。

假设用户的短期意图可以从他最近的L次交互中得出，且每个项目都可以用一个d维的嵌入向量所表示。让$X \in \mathcal{R}^{N\times d}$代表整个项目集的嵌入表示，那么将最近的L个项目依次堆叠到一起可以得到以下矩阵，其表现形式如公式1所示：

$X^u_t= \begin{bmatrix} X_{(t-L+1)1} & X_{(t-L+1)2} & ... & X_{(t-L+1)d} \\ \vdots & \vdots & \vdots &\vdots \\ X_{(t-1)1} & X_{(t-1)2} & ... & X_{(t-1)d} \\ X_{t1} & X_{t2} & ... & X_{td} \end{bmatrix} \tag{1}$

其中，最近的L个项目是$\mathcal{H}^u$的子集，用户u在时间步t的*query,key,value*在自注意力机制中都等于$X^u_t$。

首先，通过共享参数的非线性变换将*query*和*key*投影到同一空间：

$Q'=ReLU(X^u_tW_Q) \tag{2}$

$K'=ReLU(X^u_tW_K) \tag{3}$

其中，$W_Q \in \mathcal{R}^{d \times d} = W_K \in \mathcal{R}^{d \times d}$是*query*和*key*的权重矩阵，*ReLU*作为激活函数为学习到的注意力机制引入非线性的部分，其中相似度矩阵计算如公式4所示：

$s^u_t=softmax(\frac{Q'K'^T}{\sqrt{d}}) \tag{4}$

输出是一个$L \times L$的相似度矩阵，而$\sqrt{d}$用于缩放点积注意力机制，它会被设置为一个较大的值去减少梯度效应（也就是避免落入sofxmax的饱和区）。在softmax层之前会使用mask来masked相似度矩阵的对角线，以避免*query*和*key*之间的高相似度匹配分数。

其次，将*value*设置为$X^u_t$不变。与其他论文中使用线性变换映射*value*不同的是，论文采用了恒等映射，这是因为在论文模型中*value*由需要学习的参数所组成，而线性或非线性的变换会增加*value*中查看实际参数的难度，但*query*和*key*由于仅仅起辅助作用所以对于变换并不敏感。

最后，相似度矩阵会和*value*进行相乘得到自注意力机制模块的加权输出，其表现形式如公式5所示：

$a^u_t=s^u_tX^u_t \tag{5}$

输出$a^u_t \in \mathcal{R}^{L\times d}$被视作用户的短期意图表示，为了学习单个的注意力表示，将L个self-attention表示的平均嵌入作为用户的时间意图，其表现形式如公式6所示：

$m^u_t=\frac{1}{L} \sum\limits^L_{l=1}a^u_{tl} \tag{6}$

与《Attention is all you need》的Transformer结构的思路相似，模型使用正弦信号来提供时间信号与信息，时间嵌入的表现形式如公式7/8所示：

$TE(t,2i)=sin(t/10000^{2i/d}) \tag{7}$

$TE(t,2i+1)=cos(t/10000^{2i/d}) \tag{8}$

##### User Long-Term Preference Modelling

为每个用户和每个项目都分配一个潜在因子，使用$U \in \mathcal{R}^{M \times d}$表示用户的潜在因子，使用$V \in \mathcal{R}^{N \times d}$代表项目的潜在因子，可以使用点积来模拟潜在因子模型中用户-项目的交互。然而，研究表明点积违反了度量函数的重要不等式性质，将导致次优解。为了避免这个问题，论文使用了欧氏距离来衡量项目-用户之间的相似度，其表现形式如公式9所示：

$\left \| U_u-V_i \right \|^2_2 \tag{9}$

##### Model Learning

###### 目标函数

在时间步$t$给定短期注意力意图和长期偏好，任务是预测时间步$t+1$时用户可能会交互的项目$\mathcal{H}^u_{t+1}$。为了保持一致性，论文使用了欧氏距离为短期和长期的影响建模，并将他们之和作为最后的推荐分数，其表现形式如公式10所示：

$y^u_{t+1}=\omega \left \| {U_u-V_{\mathcal{H}^u_{t+1}}} \right \|^2_2 + (1-\omega)\left \|m^u_t-X^u_{t+1}\right \|^2_2 \tag{10}$

其中，第一项表示用户$u$与下一个项目$\mathcal{H}^u_{t+1}$之间的长期推荐分数，第二项表示用户$u$与其下一项之间的短期推荐分数。其中$V_{\mathcal{H}^u_{t+1}}$与$X^u_{t+1}$都是代表下一项的嵌入向量，而$V$和$X$是两个不同的参数，最后的分数结果是它们与控制因素$\omega$的加权和。

为了模型能够捕获序列中的跳过行为，模型应该能够对未来的几个项目进行预测。$\mathcal{T}^+$表示用户喜欢的未来的T个项目，$\mathcal{T}^-$表示用户不喜欢的T个项目，$\mathcal{T}^-$是从集合$\mathcal{I}/\mathcal{T}^+$中进行采样的。为了鼓励区分正用户项目对和负用户项目对，论文使用以下的基于边际的损失函数，其表现形式如公式11所示：

$\mathcal{L}(\ominus)=\sum\limits_{(u,i)\in \mathcal{T}^+}{\sum\limits_{(u,j)\in \mathcal{T}^-}{[y^u_i+\gamma-y^u_j]_++\lambda\|\ominus\|^2_2}} \tag{11}$

其中$\ominus=\{X,V,U,W_Q,W_K\}$代表模型的参数，$\gamma$是分隔正负对的边距参数。论文使用$\mathcal{l}_2$损失函数来控制模型复杂度，dropout可以应用于自注意力模块的非线性层。由于使用了欧氏距离，对于稀疏数据集可以使用norm clipping strategy将$X,V,U$约束在一个单位欧几里得球中，其表现形式如公式12所示：

$\left\|X_*\right\|_2\leqslant1,\left\|V_*\right\|_2\leqslant1,\left\|U_*\right\|_2\leqslant1 \tag{12}$

正则化方法对于稀疏数据集很有用，可以缓解curse of dimensionality problem，并可以防止数据点传播的太广泛。

###### Optimization and Recommendation

论文使用自适应梯度算法优化了所提出的方法，该算法可以自动适应步长。在推荐阶段，候选项目根据公式10计算的推荐分数按升序排列，并将排名靠前的项目推荐给用户。

模型的架构如下图所示，它包含了用户的短暂的意图和长期的偏好，通过两者相加已生成最终的推荐列表。前者是通过自注意力网络从最近的交互操作中推断出来的，整个系统是在度量学习的框架下构建的。

![image-20210702212150716](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301540137.png)

### Conclusion

本文提出了一种基于自注意力机制的度量学习方法来用于顺序推荐，模型结合了用户的短期意图和长期偏好来预测用户的下一步行动，利用了自注意力机制从用户最近的行为中学习用户的短期意图，模型可以准确地捕捉用户最近操作的重要性。此外，论文将自注意力机制推广到了序列预测任务的度量学习方法中，该方法在序列推荐中具有良好的效果。
