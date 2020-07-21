# Day1-赛题理解

## 前言

​	这也是我第一次参加这种形式的学习，像这种比赛，之前也在kaggle里有过尝试，但是都没做完过，希望这一次，在相互督促下，能够做完并尽力做的更好。理论部分看了一些，但是具体code写的挺少，也是一个需要加强的地方吧。

​	关于nlp，我目前主要针对机器翻译学习过word2vec、带有attention的seq2seq，而transform也看过一些视频，问题在于，似乎很容易忘，感觉一段时间不去看，当时看的transform已经忘了很多。

## 赛题理解

* 赛题名称：零基础入门NLP之新闻文本分类([赛题连接](https://tianchi.aliyun.com/competition/entrance/531810/introduction))
* 赛题目标：通过这道赛题可以引导大家走入自然语言处理的世界，带大家接触NLP的预处理、模型构建和模型训练等知识点。
* 赛题任务：赛题以自然语言处理为背景，要求选手对新闻文本进行分类，这是一个典型的字符识别问题。

## 解决思路

从题目就可以看到，这是一道**文本分类**的题目，我先就此找了一些相关的介绍
[NLP中关于文本分类问题的常用方案]('https://blog.csdn.net/u012155582/article/details/79688176') 

[文本分类基本流程介绍](https://www.zhuanzhi.ai/document/c65db835dfa586492a0a9141e14d2836)

这里对文本分类作了基本的说明:   语言模型、(特征降维)、分类算法。语言模型用以提取特征
![img](https://cdn.zhuanzhi.ai/images/wx/e9b956ae9f7b1ac6ba348d931cd2ad53)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图1-特征提取方法及比较</center> 

这么多种特征提取的方法，我目前使用过的实际上也只有word2vec而已...在TensorFlow中更是直接调用即可，一方面是方便了很多，另一方面也确实使得理论基础可能会相对薄弱。

![img](https://cdn.zhuanzhi.ai/images/wx/abe10df3d0360a149bd96532a472b6c0)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图2-传统机器学习的分类方法及比较</center> 

另外，在DeepLearning中，则又有如下几种解决方案:

1. fastText

2. TextCNN

3. TextRNN

4. RCNN

5. 分层注意网络（Hierarchical Attention Network）

6. 具有注意的seq2seq模型（seq2seq with attention）

7. Transformer("Attend Is All You Need")

8. 动态记忆网络（Dynamic Memory Network）

9. 实体网络：追踪世界的状态



我用过的也只有第6种而已了。

这几种模型越来越复杂，相对来说对于量比较大的数据集，训练效果也越来越好。

下图展示了某博主贴出的训练效果

![img](http://mmbiz.qpic.cn/mmbiz_png/waLJGrhJM0c8KuJO2UWIeaddypJJGsiaXXAHEus5U8icFsbPZLNpvl6RZiclIM8j8nehINcpNXTtAYuUOwrnSvlJg/0?wx_fmt=png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">（多标签预测任务，要求预测能够达到前5，300万训练数据，满分：0.5）</center> 



而DataWhale里目前给出了以下几种解决思路:

### **解题思路(DataWhale提供)**

赛题思路分析：赛题本质是一个文本分类问题，需要根据每句的字符进行分类。但赛题给出的数据是匿名化的，不能直接使用中文分词等操作，这个是赛题的难点。

因此本次赛题的难点是需要对匿名字符进行建模，进而完成文本分类的过程。由于文本数据是一种典型的非结构化数据，因此可能涉及到`特征提取`和`分类模型`两个部分。为了减低参赛难度，我们提供了一些解题思路供大家参考：

* 思路1：TF-IDF + 机器学习分类器

直接使用TF-IDF对文本提取特征，并使用分类器进行分类。在分类器的选择上，可以使用SVM、LR、或者XGBoost。

* 思路2：FastText

FastText是入门款的词向量，利用Facebook提供的FastText工具，可以快速构建出分类器。

* 思路3：WordVec + 深度学习分类器

WordVec是进阶款的词向量，并通过构建深度学习分类完成分类。深度学习分类的网络结构可以选择TextCNN、TextRNN或者BiLSTM。

* 思路4：Bert词向量

Bert是高配款的词向量，具有强大的建模学习能力。



接下来，我会跟着这几种解决思路逐渐地学习、掌握这几种方案。