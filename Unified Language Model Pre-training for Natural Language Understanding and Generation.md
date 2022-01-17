# Unified Language Model Pre-training for Natural Language Understanding and Generation

### UNILM：用于自然语言理解和生成的统一预训练语言模型

###### Published in: Advances in Neural Information Processing Systems 32 (NeurIPS 2019)

###### Paper address: https://proceedings.neurips.cc/paper/2019/hash/c20bb2d9a50d5ac1f713f8b34d9aac5a-Abstract.html

### Abstract

论文提出了一种新的统一预训练语言模型UNILM，可以针对自然语言理解和生成任务进行微调。该模型使用三种类型的语言建模任务进行预训练：单向、双向以及序列到序列预测，它使用共享的Transformer网络并利用特定的注意力掩码去控制预测条件的上下文。其在多个任务上的性能相比已有模型得到了有效的提高，与BERT的性能不相上下。

### Introduction

预训练的LM(Language Model)使用大量文本语料库数据，根据上下文预测单词的方式来学习上下文化的文本表示，其结果可以通过微调来适应下游任务。如下表所示，不同类型的LM使用不同的顺序来进行进行任务的预测和目标的训练。BERT使用的双向Transformer来融合左右的上下文去预测被[MASK]的词语，这种方式能够显著的提高自然语言理解(NLU)任务的性能，但是它的双向性导致了它很难被用于自然语言生成(NLG)任务。

![image-20210722112014950](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301540603.png)

论文提出了一种新的统一预训练语言模型UNILM，可以同时应用于NLU与NLG任务。UNILM是一个多层的Transformer网络，在大量的文本上进行预训练，优化了三种类型的无监督语言建模目标，如下表所示。

![image-20210722112645009](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301540376.png)

论文设计了一组根据上下文[MASK]预测的完形填空任务，这些完形填空任务在上下文的定义上有所不同。对于从左到右的单向LM，要预测的[MASK]的上下文由其左侧的所有单词组成。对于从右向左的单向LM，上下文由右侧的所有单词组成。对于双向LM，上下文由左右两边的单词组成。对于序列到序列LM，第二序列（目标序列）中待预测单词的上下文由第一序列（原序列）中的所有单词和目标序列中其左侧的单词组成。

与BERT类似，预先训练的UNILM可以进行微调以适应各种下游任务。但与主要用于自然语言处理任务的BERT不同，UNILM可以使用不同的自注意力掩码为不同类型的LM聚合上下文，因此可以用于自然语言处理和自然语言处理任务。

论文提出的UNILM有三个主要优点：

- 统一预训练程序生成了一个单一的Transformer，其为不同类型的LM使用共享的参数和架构，减轻了不同类型的LM的训练需要
- 参数共享使得学习到的文本表示更为的通用，因为它们针对了不同的语言建模目标进行了联合优化，上下文以不同的方式使用，减轻了对单个LM任务的过度拟合
- UNILM可以作为seq2seq的LM来使用

### Unified Language Model Pre-training

给定输入序列$x=x_1...x_{|x|}$​，UNILM可以获得每个token的上下文化向量表示，并对单向语言模型LM、双向LM和序列到序列LM进行了共享Transformer的建模优化。为了控制对要预测单词标记的上下文访问，UNILM使用了不同的mask来进行调整自注意力机制的关注范围。

![image-20210722143617252](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301540716.png)

##### Input Representation

输入$x$是一个word序列，输入遵循BERT的表示，每个token的向量表示由token嵌入、位置嵌入和段嵌入求和来计算，其中不同的LM中会使用不同的段嵌入部分。使用[SOS]标记表示输入的开头，使用[EOS]表示输入的结尾。[EOS]在NLU任务中标记句子边界，在NLG任务中学习何时终止解码过程。

##### Backbone Network: Multi-Layer Transformer

首先，输入向量$\{\mathbf{x}_i\}^{|x|}_{i=1}$​会被打包为一个矩阵$\mathbf{H}^0=[\mathbf{x}_1,...,\mathbf{x}_{|x|}]$​，然后会使用*L*层的Transformer$\mathbf{H}^l=Transformer_l(\mathbf{H}^{l-1}),l\in[1,L]$​将其编码为不同层次的上下文表示$\mathbf{H}^l=[\mathbf{h}^l_1,...,\mathbf{h}^l_{|x|}]$​。在每个Transformer变换块中，会是使用多个自注意力头来聚合前一层的输出向量。第*l*层的第$\mathbf{A}_l$​个自注意力头输出的计算方式如公式1、公式2和公式3所示：
$$
\mathbf{Q}=\mathbf{H}^{l-1}W^Q_l,\mathbf{K}=\mathbf{H}^{l-1}W^K_l,\mathbf{V}=\mathbf{H}^{l-1}W^V_l \tag{1}
$$
$$
\mathbf{M}_{ij}=\begin{cases} 0, & allow\ to\ attend \\ -\infty, & prevent\ from\ attending \end{cases} \tag{2}
$$

$$
\mathbf{A}_l=softmax(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}+\mathbf{M})\mathbf{V}_l \tag{3}
$$

论文使用了不同的mask矩阵控制token在计算上下文表示时能够处理的上下文，即当前token可以注意到的范围。

##### Pre-training Objectives

论文为不同的语言建模目标设置了四个完形填空任务，使用特殊标记[MASK]随机替换一些单词标记，而后使用softmax分类器对Transformer的输出向量进行分类对mask的token进行预测，使用交叉熵损失函数最小化预测token和原始token的损失。

###### 单向LM

在单向LM的任务中，当前token仅能从其左侧获取到信息，即只能对其左部的token和自身进行编码。可以通过一个三角矩阵来实现单向的mask，自注意力的上三角形被设置为$-\infty$，其他部分被设置为0。这样，可以使得模型仅能关注到0的部分而忽视$-\infty$​​的部分，即仅仅关注于当前以及当前以前的token，实现了单向的LM的表示。

###### 双向LM

在双向LM的任务中，当前token能够关注到其两侧的信息，即可以对整个句子中的任意部分进行编码。双向LM的mask矩阵是一个全0矩阵，模型能关注到句子中高的所有部分，实现了双向LM的表示。

###### Seq2seqLM

在seq2seqLM的任务中，句子被分为两个部分：源序列部分和目标序列部分。

在源序列部分中，当前位置的token可以关注到整个源序列部分的所有token，但不能关注到目标序列部分的任何内容，所以此部分被编码为一个正方形的全0子矩阵，该子矩阵位于矩阵的左上角。

在目标序列部分中，当前位置的token可以关注到整个源序列部分的所有token以及该token前方的目标序列token，但不能关注当前token后方的token内容，所以此部分在目标序列部分中被编码为一个三角子矩阵，该子矩阵位于矩阵的右下角。

在训练过程中，源序列和目标序列被视作一个连续的文本序列，并通过随机mask来鼓励模型学习这两个片段之间的关系，并同时训练双向的编码器和单向的解码器。

##### Pre-training Setup

总体训练目标是上述不同类型的LM目标的总和。在一个训练batch中，1/3的时间使用双向LM作为目标，1/3的时间使用seq2seqLM作为目标，1/6的时间使用从左到右的单向LM作为目标，1/6的时间使用从右到左的单向LM作为目标。

Token的mask概率为15%。在mask的位置中，80%的时间用[MASK]替换token，10%的时间用随机token替换token，其余时间保留原始token。此外，80%的时间每次随机屏蔽一个标记，20%的时间随机屏蔽两个或三个标记。

### Conclusion

论文提出了一种新的统一预训练语言模型UNILM，可以针对自然语言理解和生成任务进行微调。UNILM基于Transformer模型，使用自注意力机制对token进行编码。

在不同的LM任务上，UNILM使用了不同的mask矩阵来调整当前token所能注意到的范围：单向LM仅能注意到当前位置以及当前位置之前的tokens；双向LM能注意到整个句子中的所有tokens；seq2seqLM中句子被分为源序列和目标序列，源序列中token仅能注意到源序列中的所有tokens，目标序列中token仅能注意到源序列中的所有tokens以及目标序列中当前token以及其之前的tokens。UNILM使用不同的mask矩阵来适应多种不同的任务，并对于多个LM目标进行了联合优化，它们之间使用了共享的参数和架构以减轻了不同类型的LM的训练需要。

UNILM相比已有的模型，在自然语言生成的任务性能上得到了有效的提升，可以作为一种更先进的seq2seq模型用于文本的生成。
