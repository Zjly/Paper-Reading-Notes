# Holistic Combination of Structural and Textual Code Information for Context based API Recommendation

# 基于上下文并融合结构和文本信息的API推荐方法

###### Published in: IEEE Transactions on Software Engineering

###### Paper address: https://ieeexplore.ieee.org/abstract/document/9409670

### Introduction

如今的软件开发过程中已经难以离开API（应用程序编程接口）的使用，而开发人员要从海量的API中寻找开发所需要的API较为的困难。因此，开发人员需要API推荐工具来帮助寻找到所需的API。通常，API推荐方法会从大型的代码库中学习显式或隐式的API使用模式，并结合用户已编写源代码的上下文来进行API的推荐。

在源代码中，包含有两种核心类型的信息：

- 结构信息
  - 控制流或是数据流
  - 可以使用图形表示来表现的程序逻辑
- 文本信息
  - 代码信息、方法名或是变量名
  - 反映了代码在自然语言中的语义

例如，在图1中第8行\$hole\$中正确的API语句应该是`hashCode=str.hashCode()`。其中，该方法的方法名"compute-HashCode"和变量名"hashCode"反映了该方法的意图。而方法体内使用了多个API来实现三个相关的程序逻辑：

- 使用reader来逐行读取文件中的内容（第3/4/5/6/11/12行）
- 计算内容的哈希码（第8行）
- 将哈希值添加到创建的list中（第2/7/9行）

![image-20210630151240888](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301539876.png)

<center>图1</center>

这些程序逻辑可以在控制和数据流图中建模，且变量名（例如"path"/"result"/"rd"/"br"/"str"/"hashCode"）有助于理解相关的结构化程序逻辑。为了有效地进行API的推荐，不仅需要结构和文本代码信息的联合视图，而且需要在控制和数据流图中整体查看相关的API使用情况。

一部分API推荐方法分别的利用了结构信息或是文本信息。方法提出了许多基于源代码语言自然性，也就是依赖统计语言模型进行代码自动完成和API推荐的方法。其中统计语言模型可以是n-gram模型或是深度学习模型等不同的模型。这些方法将代码视作一系列的文本标记，并未用到源代码的结构信息，因此会因为序列长度的限制而无法正确的建模API之间的长期依赖关系。

另一部分API推荐方法利用了API的控制和数据流图，这些方法通常对控制和数据流图的子图枚举进行推荐，对程序逻辑的整体视图的利用较少。在图2中，展示了该代码片段的九个控制和数据流子图。不同的子图会被独立处理相关的API，由于较小的子图比较大的子图出现的频率更高，所以局部的较小子图内包含的API会比整体的较大子图的API更容易被推荐。

![image-20210630161826260](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301539695.png)

<center>图2</center>

论文提出了一种名为APIRec-CST的API推荐方法，解决了结构和文本代码独立建模的局限性以及对代码结构缺乏整体推理的问题。APIRec-CST是一种深度学习模型，基于Code Token Network和API Context Graph Network来结合API流程图和文本代码信息。使用上下文图为整个方法的控制和数据流图中的API使用建模，包含API推荐位置周围源代码中API使用的整体语义，API上下文图网络从中可以学习提取用于API推荐的信息结构特征。源代码中的文本代码信息（方法名、参数名或变量名）会被处理成a bag of token输入Code Token Network并与API Context Graph Network共同推断开发者的意图。

### Motivation

从图1包含的程序代码中可以抽象出多个数据流子图，其反映了部分的程序语义逻辑。例如，图2中第七个子图表示了创建读取文件的阅读器的语义，第五个子图表示了逐行阅读内容的语义。但是，所有的子图都不能获取到需要被提供的内容（计算字符串的哈希码）的语义。

![image-20210630182004527](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301539663.png)

<center>图3</center>

论文使用了API上下文图（图3）来代替控制和数据流图来表示API的使用。API上下文图是一个有向图$(N,E)$，其中N是一组节点，$E \subseteq N \times N$是一组边。N可以代表API方法调用、API字段访问、变量声明、赋值、控制单元或控制孔。E代表两个节点之间的一种流关系（控制流和数据流）。API上下文图不仅包含子图中的所有语义，还将这些语义集合成一个整体。

其中*String*类型的变量"str"只是用来存储文件的内容，在semantics-1中没有被使用，而*int*变量"hashCode"没有在semantics-2中赋值。另外，缺少了连接semantics-1和semantics-2的API，使程序逻辑完整，可以从中推断出\$Hole\$的语义是对*String*变量的某种处理，得到一个*int*类型的值。

![image-20210701110713830](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301539282.png)

<center>图4</center>

在图4中，代码的结构逻辑与图1一致，但是它们在\$Hole\$处所使用的API是不同的。但是由于它们的代码逻辑相同，仅从API上下文图中无法区分两个代码片段不同的推荐目的。图1与图4的不同之处在于代码中的文本信息，在图1中，方法名"computeHashCode"和变量名"hashCode"暗示对String类型的变量处理可能与哈希码的处理有关；在图4中，方法名"getIntegerScore"和变量名"score"暗示对String类型的变量处理可能与字符串-整型转换有关。因此，方法需要利用到代码中的文本信息，才可以完成对于空缺代码块上的正确补全。

综上所述，有效的API推荐需要结构和文本代码信息的联合视图以及整个方法的控制和数据流图中相关API使用的整体视图。

### Background

论文使用图神经网络（GNNs）中的门控图神经网络（GG-NNs）来进行API的推荐，其中节点代表API，边代表控制或数据流。此外，节点和边可以被上下文内容所标记，比如节点可以用API调用来标记，边标签可以用来区分控制流和数据流。

##### GNN

在GNN中，图的每个节点在神经网络中被表示为一个对应的单元，单元之间的连通性与图中节点的连通性相同。单元会捕获节点的当前状态，并用于在激活时计算节点的下一个状态，单元更新他们的状态并交换信息，直至它们达到稳定的平衡。节点的状态由节点标签、其传入和传出边的标签以及具有参数函数的邻居节点的状态和标签组成。节点n第t次迭代时的状态$\bold{x}_n(t)$的定义如公式1所示：

$\bold{x}_n(t)=\bold{f}_w(\bold{l}_n,\bold{l}_{co[n]},\bold{x}_{ne[n]}(t-1),\bold{l}_{ne[n]}) \tag{1}$

其中，$\bold{f}_w$是参数函数，$\bold{l}_n$是节点n的标签，$\bold{l}_{co[n]}$是包含节点n的边的标签，$\bold{x}_{ne[n]}(t-1)$是在第t-1个阶段时节点n相邻节点的状态，$\bold{l}_{ne[n]}$是节点n相邻节点的标签。这样，每个节点都可以得到一个节点表示。

![image-20210703105123647](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301539908.png)

<center>图5</center>

以图5中的图形为例，节点1在时间t的状态$\bold x_1$计算为$\bold x_1(t)=\bold{f}_w(\bold{l}_1,\bold{l}_{(1,2)},\bold{l}_{(1,3)},\bold{l}_{(1,4)},\bold{x}_2(t-1),\bold{x}_3(t-1),\bold{x}_4(t-1),\bold{l}_2,\bold{l}_3,\bold{l}_4)$，其中$\bold{l}_1$是节点1的标签，$\bold{l}_{(1,2)},\bold{l}_{(1,3)},\bold{l}_{(1,4)}$是节点1相连的边的标签，$\bold{x}_2(t-1),\bold{x}_3(t-1),\bold{x}_4(t-1)$是节点1相邻节点（节点2，节点3和节点4）在时间t-1的状态，而$\bold{l}_2,\bold{l}_3,\bold{l}_4$是节点1相邻节点的标签。节点的状态与图中的其他节点相连，因为节点可以基于信息扩散机制相互通信。

##### GG-NN

GG-NN是基于GNN的一种图神经网络，不同的是，GNN使用Almeida-Pineda算法计算梯度，而GG-NN使用门控循环单元（Gated Recurrent Units）进行时间反向传播来计算梯度。GG-NN使用了软注意力机制来决定哪些节点与计算图的最终向量表示更相关，软注意力机制以节点的状态作为输入，通过神经网络和基于在训练过程中更新神经网络参数的sigmoid函数计算每个节点的权重，对预测正确API做出更多贡献的节点赋予更高的权重。图级表示向量$\bold{x}_g$的计算如公式2所示：

$\bold{x}_g=tanh\left ({\sum\limits_{n \in N}\sigma ({i(\bold x_n(t),\bold l_n})\odot tanh(j(\bold x_n(t),\bold l_n)}\right )\tag{2}$

其中，$\sigma ({i(\bold x_n(t),\bold l_n})$作为软注意力机制工作，i和j是神经网络，将$\bold x_n(t)$和$\bold l_n$的串联作为输入并输出real-valued向量，$\odot$是逐元素乘法。

为了获得图形表示，GNN需要创建一个虚拟超级节点，该超级节点通过特殊类型的边连接到所有的其它节点，但这么做可能会破坏源代码本身的结构代码信息。此外，软注意力机制可以确定API上下文图中哪些节点对API推荐更重要。在GG-NN中，图的最终表示是每个节点的累积信息，其重要性通过软注意力机制计算，这样，一个图的最终表示就是所有节点的整体表示。

### Approach

##### Program Representation

###### 节点类型与边类型

给定一个包含\$hole\$的程序，APIRec-CST会首先构造一个API上下文图和一个代码tokens，表1展示了如何标记每种类型的节点。

- 节点的类型是一个通过API方法调用初始化的变量声明，则把API方法调用作为节点的标签
- 节点的类型是用API字段访问初始化的变量声明，则把API字段访问作为节点的标签
- 赋值语句的处理方式与变量声明语句的处理方式相同

<center>表1</center>

![image-20210703114144461](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301540459.png)

使用一个hole节点来表示\$hole\$也就是代码的缺失，当且仅当满足下列条件之一时，存在边$(n,n')\in E$：

- 从n到n'有一个直接的控制流
- 从n到n'有一个直接的数据流
- n'是一个hole节点且n是代表程序中前语句的节点
- n是一个hole节点且n'是代表程序中后语句的节点

给定边$(n,n')$，其代表n是n'的父节点，n'是n的子节点。API上下文图中的边通过用不同的类型标记区分：

- 控制流：有直接的控制流，没有直接的数据流（类型c）
- 数据流：有直接的数据流，没有直接的控制流（类型d）
- 控制和数据流：有直接的控制流和数据流（类型cd）
- 特殊流：源节点或目标节点是hole（类型s）

其中，特殊流边可以确保hole节点连接到上下文。

###### API上下文图构建

首先，APIRec-CST会构建程序的AST（抽象语法树），然后它按照以下方式基于AST在API上下文图中为程序的每个语句创建节点和边：

- 如果语句是API方法调用、API字段访问、变量声明或复制，则根据表1对应的节点类型创建节点。如果API方法调用的参数也是API方法调用或是API字段访问，APIRec-CST首先会为参数创建一个节点

- 如果语句是包含多个API方法调用或API字段访问的表达式，APIRec-CST会为每个API方法调用或API字段访问创建一个节点

- 如果语句是一个控制语句，APIRec-CST根据其类型为控制单元创建一个节点，如表2所示。例如，如果当前语句是一个while语句，APIRec-CST会首先创建一个while节点，一个condition节点和一个body节点，引入了两条从while节点到condition节点和while节点到body节点的类型c的边

<center>表2</center>

![image-20210703150103934](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301540604.png)

然后，根据节点之间的数据依赖关系构建边。例如，while语句中，从condition节点到第一个条件控制表达式节点创建一条类型c的边，再从body节点到循环体内的第一个节点创建一条类型c的边，最后从while节点到循环体外的节点创建一条类型c的边。如果程序包含hole节点，那就对hole节点的前后语句都创建一条类型s的边。

为了分析数据依赖关系，如果一个节点表示一个变量声明语句或赋值语句，这意味着这个节点包含一个变量或一个对象，可以在其他API调用中用作接收器或参数，那么我们存储它的变量或对象。然后，如果一个节点的存储变量或对象被用作另一个节点中表示API调用的接收器或参数，则从前一个节点到后一个节点之间创建一条类型为d的边。如果这两个节点之间存在控制流，则在这两个节点之间创建一条cd类型的边。

###### 代码tokens构建

代码tokens由方法名、参数名和变量名的tokens组成。首先，APIRec-CST会根据AST提取出方法名、参数名和变量名。然后，由于开发人员经常使用复合词或非标准词作为名称，提取的名称会被拆分为tokens：

- 去除名字中的数字，拆分数字两端的word
- 根据特殊字符拆分组合单词
- 根据驼峰命名拆分单词
- 删除无意义的单词

##### Architecture

APIRec-CST由两个主要部件组成：API Context Graph Network和Code Token Network，通过一个联合层将它们的输出连接在一起，其结构图如图6所示。API Context Graph Network基于给定的API上下文图学习API上下文图向量，Code Token Network由嵌入层和GG-NN组成；根据给定的代码tokens学习tokens向量，由嵌入层、隐藏层和求和运算组成。联合层旨在结合API上下文向量和tokens向量并输出联合向量，然后使用softmax函数根据联合向量计算每个候选API的概率。

![image-20210703165234537](https://raw.githubusercontent.com/Zjly/Image-hosting/master/202112301540964.png)

<center>图6</center>

###### API Context Graph Network

API Context Graph Network将API上下文图作为输入并输出一个向量，API上下文图被处理为一组节点和边，并输入到网络中 。首先使用嵌入层将每个节点的标签嵌入到一个单独的向量之中，然后将其用作GG-NN中节点注释的初始向量。嵌入层维护一个嵌入矩阵，将每个节点标签映射到一个单独的向量之中，并通过训练更新嵌入矩阵。GG-NN首先计算每个节点的状态，并将最后一个时间步的状态作为节点表示。然后基于节点表示计算API上下文图向量，使用软注意力机制来决定节点的权重。

###### Code Token Network

Code Token Network将tokens作为输入并输出一个向量。首先使用嵌入层将代码token嵌入到单独的向量之中，每个token的信息以向量的形式进行编码，可以在训练中进行优化学习。在模型中，忽略了tokens的顺序并把它们视作一个词袋。将最后一个隐藏层的输出的每个token的所有向量表示相加，作为最后的输出。

###### Joint Layer

联合层将API上下文图向量和token向量作为输入，输出联合向量。假设API上下文图向量是一个$d^A$维向量，token向量是一个$d^T$维向量，联合层将它们拼接为$d^{A+T}$维向量。然后，联合向量通过*tanh*作为激活函数的全连接层，可以以整体的方式进一步学习结构代码信息和文本代码信息的联合语义。

###### Softmax Function

Softmax 函数将联合向量作为输入，输出所有API的归一化概率。

### Conclusion

论文提出了一种基于深度学习的API推荐方法，将API usage与源代码中的文本信息相结合，同时学习结构和文本特征。 评估表明，论文的方法明显优于现有的基于图的统计模型和基于树的API推荐深度学习模型，可以更快、更准确地完成编程任务。 
