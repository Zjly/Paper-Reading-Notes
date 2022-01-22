# Improving Language Understanding by Generative Pre-Training

### GPT：一种生成式预训练的语言模型

### Introduction

在大多数自然语言处理的任务中，都需要大量的已标注的信息来完成模型的训练，但并不是所有的任务都能够获得足够的已标注的信息，这也就影响了模型的效果。事实上，随着如今模型的参数的不断增加，大多数任务的数据集的大小并不能满足模型训练的需求，因此通过预训练的方式来初始化模型的参数权重成为了被普遍的使用的方法之一。

GPT模型使用了一种生成式的预训练方法来得到一种通用的模型表示，即模型使用大规模的无监督语料库来预训练word embedding，使用小规模的有监督的数据集进行微调，这种方式也成为了现在模型的主流训练方式。

### Framework

##### Unsupervised pre-training

GPT使用语言模型来进行预训练，并使用了n-gram方法对当前单词进行预测。通俗的说，也就是根据前k个单词$u_{i-k},...,u_{i-1}$来预测单词$u_i$具体为哪一个单词，并最大化单词出现的可能性$P$，$P$是根据参数为$\Theta$的神经网络所建模的。其数学表示如公式1所示，使用了负对数最大似然的损失函数来计算loss：

$L_{1}(\mathcal{U})=\sum\limits_{i}\log P(u_i|u_{i-k},...,u_{i-1};\Theta) \tag{1}$

GPT与BERT都使用了Transformer作为模型的基础，这种由全attention所组成的模型在各种任务上都取得了SOTA的效果，逐渐替换了RNN、LSTM等经典架构的模型，成为了如今的热门之选。与BERT使用了Transformer的编码器不同的是，GPT使用了Transformer的解码器作为了模型的结构。其数学表示如公式2所示：

$\begin{gather} h_0=UW_e+W_p \\ h_l=transformer\_block(h_{l-1})\forall i\in[1,n] \tag{2} \\ P(u)=softmax(h_nW_e^T) \end{gather}$

其中，$U=(u_{-k},...,u_{-1})$代表token的上下文向量，$W_e$是token的embedding矩阵，$W_p$是position embedding的矩阵。$h_0$表示token的word embedding与position embedding之和，即每一个token由单词含义部分与位置部分所组成。

#####  Supervised fine-tuning

在得到预训练模型之后，就可以以预训练模型为基础，在已标注的数据集上进行微调。数据集$\mathcal{C}$的格式形如$x^1,...,x^m->y$，其中$x^1,...,x^m$为token，$y$为标签。数据经过预训练模型之后，输入到softmax层进行分类，得到模型所预测的结果。其数学表示如公式3所示：

$P(y|x^1,..,x^m)=softmax(h^m_lW_y) \tag{3}$

同样，在微调部分也使用了负对数最大似然损失函数计算loss，如公式4所示：

$L_2(\mathcal{C})=\sum\limits_{(x,y)}P(y|x^1,..,x^m) \tag{4}$

修改embedding部分的参数作为微调部分的辅助目标可以具有提高模型泛化程度和加速收敛的作用，因此loss为预训练部分的loss与微调部分的loss的带权和，如公式5所示：

$L_3(\mathcal{C})=L_2(\mathcal{C})+\lambda L_1(\mathcal{C}) \tag{5}$

##### Task-specific input transformations

对于文本分类任务来说，在原模型后加入softmax层即可以获得分类的结果，但是对于文本蕴涵、相似度检测等NLP任务时，简单的softmax层就无法有效地完成任务。因此，GPT使用了traversal-style方法，将结构化输入转换为预先训练好的模型可以处理的有序序列，如图所示：

![image-20220122222016283](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202201222222341.png)

###### 文本蕴涵

连接前提P和假设H的token序列，并使用分隔符D来分隔P和H，Transformer模型的输出结果输入线性层。

###### 相似度检测

使用两种不同的顺序（句子1/句子2、句子2/句子1）来连接它们的token序列，并使用分隔符来分隔句子1和句子2，对Transformer模型的输出结果进行元素加操作后输入线性层。

###### QA与常识推理

连接问题Q和答案A的token序列，将每一组Q与A单独的连接为一条数据，并使用分隔符来分隔问题Q和答案A，Transformer模型的输出结果输入线性层独立处理，然后通过softmax层进行规范化，以在可能的答案上产生输出分布。

### Conclusion

GPT模型使用预训练-微调的结构，基于Transformer构建了一个在NLP上有着卓越成果的模型，为之后的GPT-2、GPT-3得出现提供了一个良好的基础。GPT作为Transformer被提出之初的模型，也算是见证了Transformer的发展与壮大。
