# Attention Is All You Need

### Abstract

在Transformer模型被提出之前，主流seq2seq结构的模型大多都是基于encoder和decoder所组成的，对其的改进更多的是在encoder-decoder结构之上加入attention机制，以此来优化模型的性能。

Transformer模型是一种完全基于注意力机制的网络模型，在其中抛弃了以往所主流使用的递归层和卷积层。Transformer仅由Multi-Head Attention和Feed Forward所组成，并可通过模块的堆叠来完成Transformer网络的搭建。

### 1 Introduction

Seq2seq模型例如RNN在语言建模和机器翻译等任务中使用十分广泛，它们主要延时间步$t$，通过上一个时间步的隐藏状态$h_{t-1}$和当前时间步$t$的输入$Input_t$，对隐藏状态$h_t$进行更新并输出。但是，这种读取方式是严格按照时间步的顺序的，这导致了在较长序列中不能进行并行化的计算，限制了模型的性能。

Attention机制被提出之后，被广泛应用到各种RNN网络之中，强化了RNN网络获取长距离的信息的能力，使得RNN能够捕获到输入序列中每一个部分的信息。Transformer模型正是完全基于attention机制，抛弃了其中的RNN部分，利用attention来捕获全局范围内的输入和输出信息。

### 2 Background

现有的模型例如Extended Neural GPU，ByteNet和ConvS2S获取长距离的信息的能力较弱，它们将两个位置的符号关联起来的操作数会随着距离的增长而增加，复杂度是线性或对数级别的，而Transformer使用了Multi-Head Attention将其降低到了常数级别。

Self-attention将单个序列的不同位置进行联系，以此计算序列的representation，Transformer是第一个完全使用self-attention的模型。

### 3 Model Architecture

对模型的解释在https://blog.csdn.net/jiaowoshouzi/article/details/89073944和https://blog.csdn.net/Urbanears/article/details/98742013中有着详细的介绍。

##### 3.1 Encoder and Decoder Stacks

Transformer模型基于encoder-decoder结构，即通过encoder将输入序列$(x_1,...,x_n)$映射到序列$z=(z_1,...,z_n)$，decoder根据$z$逐个生成输出序列$(y_1,...,y_m)$，其中在每个时间步中模型都是auto-regressive的，模型的结构图如下所示。

![image-20210528195805099](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538555.png)

Encoder部分由6个相同的模块组成，其中每个模块都由两个部分所组成：Multi-Head Attention和Feed Forward。每个子层都使用了残差连接和layer normalization，也就是说每个子层的输出都是$LayerNorm(x+Sublayer(x))$，为了方便进行残差连接，模型中所有子层和embedding层都会生成维数为512的输出。在模型中使用了残差连接，能够更好地解决网络退化的问题，使得信息能够更好地传递，且模型的性能不会受到大的影响。使用layer normalization对单个样本的不同特征进行了归一化，将其满足于均值为0方差为1的分布，避免了数据落在饱和区。

Decoder部分同样由6个相同的模块组成，其中每个模块由三个部分组成：Masked Multi-Head Attention，Multi-Head Attention和Feed Forward。与encoder不同的是，其中加入了一层Multi-Head Attention用于对encoder的输出进行attention的操作。同时，decoder将第一层Multi-Head Attention加入了mask，以此来限制decoder获取的信息，使其只能获取到当前位置及之前位置的信息。

##### 3.2 Attention

Attention机制将向量query，key和value映射到向量output，其中output是value的加权和，其权重由query和key计算得到。

###### 3.2.1 Scaled Dot-Product Attention

Attention机制在Transformer模型中的结构如下图所示，输入由query，key和value所构成，其中query和key的输入维度为$d_k$，value的输入维度为$d_v$。权重计算方式为：首先计算query和key的点积，之后将点积结果除以$\sqrt{d_k}$，之后使用softmax函数获得最后的结果也就是value的权重。

![image-20210529181722781](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538340.png)

实践中，Transformer模型将query，key和value合并成矩阵Q，K和V便于并行计算，attention的结果如下公式1所示：

$$Attention(Q,K,V)=softmax(\frac {QK^T}{\sqrt{d_k}})V \tag{1}$$

$QK^T$这部分主要是对于两个矩阵进行相似度的计算，也就是获取到句子中每个部分对于其余部分的关注度。Q代表了查询值，需要使用这个查询值与每个K也就是key进行对比进行相似度的计算，V作为value保留了输入的特征。

两种常用的attention函数为additive和点积attention，Transformer模型使用了点积attention，由于使用了矩阵乘法，点积attention的时间复杂度和空间复杂度都要优于additive attention。

对于数值较小的$d_k$，两种attention函数的表现效果较为的类似。但是在$d_k$较大时，由于点积在数量级之上增长很大，结果会处于softmax函数的梯度较小的区域也就是饱和区，其性能会不如additive。为了解决这个问题，Transformer使用了比例因子$\frac{1}{\sqrt{d_k}}$来放缩点积的值。

###### 3.2.2 Multi-Head Attention

使用不同的投影函数投影Q，K和Vh次，将它们分别投影到$d_q$，$d_k$和$d_v$相比使用单独的$d_{model}$维的函数投影的效果要好。在每个Q，K和V的投影上都并行的执行attention函数，都产生$d_v$维的输出，它们会进行一个concat操作后得到最终值，结构如下图所示。

![image-20210529204502464](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538077.png)

Multi-Head Attention能够使得模型在不同的位置共同关注到不同的表示子空间的信息，相比仅使用一个attention head，这种结构能够更好地获取到更多方面的信息，在某个位置上关注到不同的关注点，加大了模型对于句子的获取能力。其公示表示如2和3所示：

$$MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O \tag{2}$$

$$head_i=Attention(QW^Q_i,KW^K_i,VW^V_i) \tag{3}$$

也就是说，每一个单独的$head_i$都有其的$Q_i$，$K_i$和$V_i$，它们是通过原始的Q，K和V进行以$W_i$为参数的线性变换所得到的。

在Transformer模型中，使用了$h=8$的head，其中$d_k=d_v=d_{model}/h=64$。此时每个head的维度相比单head都有所减小，总体的计算量和单head所相似。

###### 3.2.3 Applications of Attention in our Model

Transformer在三个地方使用了multi-head attention：

![image-20210529211108862](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538843.png)

+ 在encoder-decoder的multi-head attention层之中，query来自于decoder的前一个层，而key和value来自encoder的输出。这使得decoder中的每一个单词都能够获取到encoder中每个位置的信息，也就是说，decoder中每一步单词的生成都能够获取到encoder句子中所有位置的信息以及权重，和原本的encoder-decoder模型的attention机制类似。

+ 在encoder中的multi-head attention层之中，query，key和value都来自encoder的前一层也就是embedding层的输出。这使得encoder中的每一个单词都能够获取到自身所在的句子中每个位置的信息，也就是说，能够获取到这个句子中每个单词在句子中的意义。

+ 在decoder中的masked nuiti-head attention层之中，query，key和value都来自decoder的前一层的输出。不同的是，decoder中的每个单词不被允许了解到整个句子中的所有单词的信息，因为其不能获知未来的单词的生成，所以在其中加入了mask来限制获取到该位置之后的单词的信息。也就是说第n个单词的后部的信息$(n+1,...)$都被设置为$-\infty$也就是masked了。

##### 3.3 Position-wise Feed-Forward Networks

  在encoder和decoder的每一层之中，都包含一个全连接前馈网络，其中包括两个线性变换，中间使用了一个RELU激活函数，其公式表示如4所示：

$$FFN(X)=max(0,xW_1+b_1)W_2+b_2 \tag{4}$$

它们在不同的地方使用不同的参数，其维度$d_{model}=512$，其隐藏层单元$d_{ff}=2048$。加入了这一步之后，能够加强模型的特征抽取能力。

##### 3.4 Embeddings and Softmax

与其他seq2seq模型相同的是，Transformer使用了训练好的embedding层去转换input和output，将其转化为一个固定长度的向量$d_{model}$，并使用softmax将输出转化为下一个输出字符的概率。在embedding层中，会将权重矩阵乘以$\sqrt{d_{model}}$，其中encoder和decoder的embedding层共享相同的权重矩阵。

##### 3.5 Positional Encoding

由于Transformer模型仅使用了attention机制，所以序列中每个位置的顺序是没有被利用到的，所以，将位置编码加入到了输入的embedding中，其维度被设置为$d_{model}$以此能够与embedding的结果所相加，其公式表示如5和6所示：

$$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}}) \tag{5}$$

$$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}}) \tag{6}$$

选择这个函数的原因主要是需要一种能够获取到位置信息的区别、且差异不依赖于文本长度的位置编码，所以使用了一种有界的周期函数。在https://www.zhihu.com/question/347678607有较为完善的解答，使用sin和cos可以保持相对位置的线性关系。

### 4 Why Self-Attention

将attention层与循环层和卷积层所比较，后两者通常被表现为encoder或是decoder中的隐藏层。

文章从三个方面来证实了Transformer的优越性：

+ 每层的总计算复杂度

+ 可以并行化的计算量，以所需的最小顺序操作数来衡量

+ 网络中远程依赖之间的路径长度

在seq2seq的任务中，长距离的信息的获取一直是一个难题。这主要是由于在一个网络中，前向信号和反向信号如果要在两个不同位置之间进行传播，不同位置之间的距离$distance$会成为它们必须经过的一段路程，这也就会影响到网络的学习能力。因此，两个不同位置之间的路径长度越短，也就会越容易学习其中的长距离依赖关系。

![image-20210530130051695](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538179.png)

Self-Attention有着常数级别的复杂度去进行序列的连接，而循环层需要$O(n)$级别的时间来连接所有的位置。

在计算的复杂性方面，self-Attention的时间复杂度为$O(n^2·d)$，而循环层的时间复杂度为$O(n·d^2)$，也就是说，当序列长度n小于维数d时，self-Attention的性能要更优秀。作者还提出了一种能够减少长序列时间复杂度的办法，也就是使用"n-gram"的思想，将self-attention限制在以自身为中心的、长度为r的领域之中。

卷积层不会连接所有的输入和输出位置，如果要增加网络中任意两个位置的最长路径长度，卷积的时间复杂度为$O(n/k)$，而dilated convolutions（空洞卷积）的时间复杂度为$O(log_k(n))$，separable convolutions（可分离卷积）的时间复杂度为$O(k·n·d+n·d^2)$。由此可见，self-Attention的复杂度要优于卷积层。

### 5 Training

##### 5.1 Training Data and Batching

文章使用Transformer模型在WMT 2014 English-German dataset和WMT 2014 English-French dataset上进行了训练，大约由450万个句子和3600万个句子所组成。

##### 5.2 Hardware and Schedule

文章在8个NVIDIA P100 GPUs上进行了训练，文章中模型进行了10万steps共12小时的训练，大型模型进行了30万steps共3.5天的训练。

##### 5.3 Optimizer

文章使用了Adam optimizer，学习率公式如7所示：

$$lrate=d^{-0.5}_{model}·min(step\_num^{-0.5},step\_num·warmup\_steps^{-1.5}) \tag{7}$$

这相当于在$warmup\_step$中线性地增加学习率，然后按步骤数的平方反比降低学习率。

##### 5.4 Regularization

文章将dropout应用于每个子层的输出并进行了归一化，并且在训练期间使用了label smoothing加入了噪声，缓解了过拟合的现象出现，使模型的泛化能力更强。

### 6 Results

##### 6.1 Machine Translation

Transformer模型在英德、英法翻译任务上的BLEU要优于之前提出的模型，并且训练使用的算力也更为的少，证明了模型的有效性和性能。

![image-20210530141012335](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538363.png)

##### 6.2 Model Variations

文章经过参数的尝试获得了一组最优的参数。在实验中，（A）表现了multi-head的数量过多也会影响到模型的质量，需要找到一个最优的head的数量。（B）表现了减少key的大小$d_k$也会影响到模型的质量，这说明了确定compatibility function并不容易，且比点积attention更加复杂的compatibility function有可能获得更好的效果。（C）和（D）表现了大型的模型的训练效果更好，且dropout缓解了过拟合的现象。（E）表现了positional embeddings和sinusoidal positional encoding的区别不大。

![image-20210530141026130](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538936.png)

##### 6.3 English Constituency Parsing

文章将模型运用到了English constituency parsing任务之中，证明了模型的泛化性能优秀。

### 7 Conclusion

Transformer模型是第一个完全基于attention机制的seq2seq模型，使用了multi-head attention替换了循环层，其性能要优于基于循环层和卷积层的模型。在翻译任务中，评价指标要好于以往的模型。

### Attention Visualizations

![image-20210530142656128](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538264.png)

这里展现了making这个词在attention机制下的关注点，可以看到making的关注点主要集中在more和difficult上，不同的头的关注重点不同，表示了在不同子空间下making的不同关注表示。关注点构成了短语making...more difficult。

![image-20210530144538478](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538827.png)

展现了its这个词在attention head5和6之下的关注点，可以发现its的关注点非常的集中，聚焦在了law和application之上，代表了这个its就代指了law。

![image-20210530144606281](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301538671.png)

展现了两个不同的head的关注点，可以看到他们的关注点并不相同，每个head能够执行不同的任务也就是说关注到句子中各个部分种不同的信息。
