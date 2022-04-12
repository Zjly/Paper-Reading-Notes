# What Is the Next Service? An Encoder-Decoder Approach for Service Composition Recommendations

###### Published in: ICWS 2022

### Abstract

如今的服务推荐技术大多基于协同过滤（用户-服务调用关系）和基于内容（用户历史行为建模），但是这些方法无法正确的建模服务功能重叠，即一些推荐的服务可能提供了相似的功能，这会导致推荐的服务组合的准确性的降低。

因此，论文提出了一种基于Encoder-Decoder结构的模型EDeR，将服务推荐问题转化为生成问题。

### Introduction

现在推荐服务组合的方法主要有两类：

- 基于语义或者adapters 来推荐服务——基于输入、输出和接口进行匹配（依赖语义信息）
- 基于之前使用的历史服务组合进行匹配——基于内容、基于协同过滤的方法

EDeR通过三个步骤来进行服务组合的推荐：

- 清理原始数据，生成服务图，并根据在所有mashup中出现的概率对服务进行排序；
- 利用图的自监督学习，得到每个服务的嵌入和每个词的嵌入；
- 基于服务嵌入和单词嵌入，使用编码器-解码器模型将用户需求映射到满足用户需求的服务序列。

论文的主要贡献：

- 将组合服务推荐问题转化为序列生成问题，根据用户需求逐个识别服务以进行推荐；
- 由于服务描述并不总是准确的，在服务图中使用自监督 学习进行服务嵌入来进一步补充。此外，在下游任务中使用嵌入，并可视化嵌入；
- 提出了一种EDeR算法，将用户的需求和类别信息编码成隐藏向量，解码成服务的概率分布，获取概率最高的服务，最后使用自回归得到一个复合服务；
- 进行了大量的实验来评估EDeR，实验结果表明它显著优于最先进的方法

### Motivation

![image-20220410170116431](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204101708579.png)

如图所示，在一个开发过程中，开发人员有一个大致的需求描述，会根据需求描述逐个寻找到服务$w_1，w_2$，即在现有的服务的基础上找到下一个服务。这种生成模式与自然语言生成和翻译的过程较为的相似，因此可以使用seq2seq的模型进行服务的生成。

### EDeR

![image-20220411194932766](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204111949576.png)

如图所示，EDeR主要分为三个部分：

- Preprocessing：EDeR对用户需求描述进行预处理，生成服务语义图，并构建mashup服务序列，为更好地嵌入用户需求和API做准备
- API Embedding：EDeR使用一种自监督学习算法来获得每个API在生成模型中使用的标签服务图中的嵌入
- Encoder-Decoder-based Recommender：利用mashup服务序列、用户需求描述和类别信息，设计了一个基于编码器-解码器的模型，以了解用户需求和组合服务之间的关系

##### Preprocessing

- 过程数据处理：对噪声数据进行预处理，删除拼写错误的单词和非信息性单词（标点删除、停止词删除、小写转换和词性还原）
- 构造图：构造服务语义图$G=(V,E)$，其中$V=T\cup A$（A???，我就没见到过A，猜测是作者手滑把S打成了A）。$T$代表标签，$S$代表服务，$E$代表标签和服务之间的边。
- 构造序列：服务的顺序在深度学习模型的收敛过程中起着重要作用。对于具有无序服务的服务mashup，很难确定应该使用什么策略对其进行排序。因此，作者根据它们在训练数据中出现的频率对它们进行排序。 

##### API Embedding

- 使用BERT通过标签或API的描述来获得它们的初始嵌入。
- 使用GraphSAGE通过无监督学习进一步整合特征，从而补充不准确服务描述的特征。 

![image-20220411205514965](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112055750.png)

如图所示，对相邻节点的嵌入进行平均，并对平均值进行非线性变换：

![image-20220411224357365](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112246694.png)

其中$h^{k}$是第$k$步的节点的表示。每个节点$v\in\mathcal{V}$聚集了直接邻域节点的表示$\{h^{k-1}_u, \forall \mu \in \mathcal{N}(v) \}$

![image-20220411224434900](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112246658.png)

其中$h^{k-1}_v$是节点的当前表示，$h_{\mathcal{N}(v)^{k-1}}$是聚合的邻域向量。

- 节点$u$及其相邻节点$v$具有类似的嵌入，但没有交互的节点$v_n$具有不同的嵌入。损失函数为： 

![image-20220411224446405](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112246366.png)

其中，$z_u$是节点$u$的嵌入，$v$是节点$u$通过随机游走得到的邻居，$v_n\sim P_n(v)$代表负采样，$Q$代表样本数。

为了确保模型的有效性，使用逻辑回归来评估嵌入，使用主类别作为基本事实来获得评估分数。

##### Encoder-Decoder-based Recommender

Encoder

- 获得序列中每个标记的可嵌入表示

- 采用bi-GRU对嵌入向量序列进行编码，其中$c$和$r$表示类别描述和用户需求描述，$d_c$和$d_n$表示类别描述和用户需求描述：

  ![image-20220411224506114](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112246297.png)

###### Decoder

- 基于$h_r$，解码器状态$s_t$用于关注用户需求描述和类别信息中的token，其中$s_{t-1}$是decoder的前一个隐藏状态，$e(y_{i-1})$是token$y_{i-1}$的向量表示，$h^{'}_r$是encoder的最后一个隐藏状态

![image-20220411224825653](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112248510.png)

- 为了整合足够的信息，采用了如图所示的分层指针网络，共同参与并复制来自用户需求描述和类别信息的tokens

  ![image-20220411225202872](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112252194.png)

- $W$、$W^{'}$、$b$和$b^{'}$是可训练参数，$P^c_t$和$P^r_t$是词汇表中所有API的概率分布

![image-20220411225634188](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112256551.png)

- 最后，得到了所有API词汇表的总体分布$P_t(w)$ 

![image-20220411230025203](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112300250.png)

其中$ω$、$ω^{'}$、$ω_x$和$b_{ptr}$是可训练参数，$y_{t−1}$为解码器输入，$g$为非线性函数，$θ_t$用于平衡用户需求描述和类别信息产生的概率

###### Attention

- 注意力上下文向量$c_t$取决于隐藏状态$h_j$和encoder隐藏状态源序列$h=(h_1，…，h_T)$的之间的相关性： 

![image-20220411230449650](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112305004.png)

![image-20220411230458815](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112305323.png)

其中$s_t$是decoder的隐藏状态，$e_{tj}$表示encoder状态$s_t$在时间步$t$处受$h_j$影响的程度。这样，decoder可以决定要注意源token的哪些部分

###### Training

使用负对数似然损失函数对模型进行训练：

![image-20220411230713530](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112308373.png)

其中$r$和$c$是用户需求描述和类别信息的源序列，并且$\mathcal{\theta}$表示所有可学习的模型参数

###### Inference

在softmax对所有API进行排序后，我们对API进行排序，并选择前k个API作为自动回归的推荐服务或供开发人员选择

![image-20220411230817318](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204112308572.png)

对于inference阶段，由于mashup数据集中的组合服务长度较短，使用贪婪算法自回归获得组合服务。

具体来说，在输入源需求描述之后，解码器逐个生成目标服务。首先，符号\<soa\>表示API的开始，它在开始时传递，并生成输出API分布。然后，直接选择概率最高的API作为结果，并将其用作下一个输入，直到结束符号\<eoa\>表示API的结束。这样，就生成了一个复合服务。 

### Experiments

![image-20220412125835044](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204121258626.png)

![image-20220412125941487](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204121259803.png)

![image-20220412125959463](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204121300657.png)

![image-20220412130316750](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204121303476.png)

![image-20220412130448080](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204121304172.png)

### ![image-20220412130502797](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202204121305029.png)Conclusion

本文提出了一种基于Encoder-Decoder的推荐器（EDeR）来解决复合服务推荐的问题。与现有方法相比，EDeR解决了服务功能重叠导致推荐不准确的问题。

- 在嵌入用户需求和API之前，处理原始数据进行准备
- 利用图的自监督学习，得到每个服务的嵌入
- 通过glove嵌入，我们得到了每个单词的嵌入
- 基于服务嵌入和单词嵌入，使用编码器-解码器模型将用户需求转换为一系列服务，以满足用户需求

实验结果表明，EDeR实现了最先进的性能。未来，将利用有关版本、其他关系等的其他信息来提高EDeR的性能。 
