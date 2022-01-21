# Improving Language Understanding by Generative Pre-Training

### GPT：一种生成式预训练的语言模型

### Introduction

在大多数自然语言处理的任务中，都需要大量的已标注的信息来完成模型的训练，但并不是所有的任务都能够获得足够的已标注的信息，这也就影响了模型的效果。事实上，随着如今模型的参数的不断增加，大多数任务的数据集的大小并不能满足模型训练的需求，因此通过预训练的方式来初始化模型的参数权重成为了被普遍的使用的方法之一。

GPT模型使用了一种生成式的预训练方法来得到一种通用的模型表示，即模型使用大规模的无监督语料库来预训练word embedding，使用小规模的有监督的数据集进行微调，这种方式也成为了现在模型的主流训练方式。

### Framework

##### Unsupervised pre-training

GPT使用语言模型来进行预训练，并使用了n-gram方法对当前单词进行预测。通俗的说，也就是根据前k个单词$u_{i-k},...,u_{i-1}$来预测单词$u_i$具体为哪一个单词，并最大化单词出现的可能性$P$，$P$是根据参数为$\Theta$的神经网络所建模的。其数学表示如公式1所示：

$L_{1}(\mathcal{U})=\sum\limits_{i}\log P(u_i|u_{i-k},...,u_{i-1};\Theta) \tag{1}$

GPT与BERT都使用了Transformer作为模型的基础，这种由全attention所组成的模型在各种任务上都取得了SOTA的效果，逐渐替换了RNN、LSTM等经典架构的模型，成为了如今的热门之选。与BERT使用了Transformer的编码器不同的是，GPT使用了Transformer的解码器作为了模型的结构。其数学表示如公式2所示：

$\begin{gather} h_0=UW_e+W_p \\ h_l=transformer\_block(h_{l-1})\forall i\in[1,n] \tag{2} \\ P(u)=softmax(h_nW_e^T) \end{gather}$

其中，$U=(u_{-k},...,u_{-1})$代表token的上下文向量，$W_e$是token的embedding矩阵，$W_p$是position embedding的矩阵。$h_0$表示token的word embedding与position embedding之和，即每一个token由单词含义部分与位置部分所组成。
