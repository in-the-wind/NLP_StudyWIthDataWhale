# Task3 基于机器学习的文本分类



对于使用机器学习的文本分类，我们可以借助sklearn中现成的模型来完成。

在观看了阿水的直播讲解后，发现作为baseline的TF-IDF只有十几行，同时可以达到0.87左右的准确率，感觉还是有点超出我的想象的。



#### **TF-IDF**

TF-IDF 分数由两部分组成：第一部分是**词语频率**（Term Frequency），第二部分是**逆文档频率**（Inverse Document Frequency）。其中计算语料库中文档总数除以含有该词语的文档数量，然后再取对数就是逆文档频率。

```
TF(t)= 该词语在当前文档出现的次数 / 当前文档中词语的总数
IDF(t)= log_e（文档总数 / 出现该词语的文档总数）
```

在sklearn中，只需要调用TfidfVectorizer即可实现这一特征的提取。



这里我参考了另一位分享的代码，展示了TfidfVectorizer,LogisticRegression,XGBClassifier三种方式的实现。其分享的代码中使用了 GridSearchCV进行超参数搜索，其会在给定范围内按照一定步长进行不断尝试，尝试。

GridSearchCV可以保证在指定的参数范围内找到精度最高的参数，但是这也是网格搜索的缺陷所在，它要求遍历所有可能参数的组合，在面对大数据集和多参数的情况下，非常耗时。这也是我通常不会使用GridSearchCV的原因，一般会采用后一种RandomizedSearchCV随机参数搜索的方法。

这几天也正在学习TensorFlow，在超参数搜索部分讲解的RandomizedSearchCV也做了一些笔记说明。

因为自己也是不那么熟悉的，所以尽可能写的会详细一些，显得啰嗦一些。

对于这里机器学习直接使用的是sklearn，第一步的转化为sklearn模型可省略，但是之后的深度学习练习中想必一定会用到的，所以都放在这里。

## 超参数搜索

对于训练一个模型而言，超参数是需要我们来调整的，如果我们手动地去设置这些超参数，每次修改后再运行，显然是非常慢、麻烦且费时的。

例如，对于单个的学习率的设置:

![image-20200721222154030](F:\tf2\image-20200721222154030.png)

这里一个超参数我们就需要一个for循环，如果还有其他的超参数，就需要自己再加多个for循环，显然是很不方便。



这里，我们通过sklearn提供的方法来进行超参数搜索，它可以通过并行的方式方便地尝试各种超参数，并能记录下最好的超参数及模型。



```python
# Randomizedsearchcv
# 1．转化为sklearn的model
# 2.定义参数集合
# 3．搜索参数
```

### 转化为sklearn的model

sklearn提供了一个`Randomizedsearchcv`的方法来进行超参数的搜索，但是必须要是sklearn的model才能调用这个方法。

因此，我们要先将TensorFlow的model转化为sklearn的model

[TensorFlow中提供的转化为scikit_learn model的接口](https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasRegressor)

![image-20200721233442552](F:\tf2\image-20200721233442552.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">分别为分类任务和回归任务各自提供了一个转化的API</center>

我们在使用时，只需根据自己的任务，选择合适的API转化即可。

为了转化为sklearn model，我们首先需要写一个build_model函数用来表示模型是如何建立的。

![image-20200721233813475](F:\tf2\image-20200721233813475.png)

![image-20200721234340903](F:\tf2\image-20200721234340903.png)

建立好了build_model后，就可以调用刚刚看的接口，生成sklearn的model。

通过调用fit，并用history记录过程中的值。

![image-20200721234705451](F:\tf2\image-20200721234705451.png)

### 定义参数集合

选择我们要进行搜索的超参数的集合。

实际上，在`build_model`中，我们已经定义了这些，如上图的

`hidden_layers`:隐藏层数目。

`layer_size`:每一层的大小。

`learning_rate`:学习率

![image-20200721235355442](F:\tf2\image-20200721235355442.png)



对于选择数目比较少的离散量，直接定义一个数组来选择；如`hidden_layers`

`layer_size`,这里设置的可能是1到100中的整数，当然也可以设置为其他

`learning_rate`，我们希望设计为连续值而不是离散量，这里我们采用`scipy`的`reciprocal`分布来生成，其表达式就是上面的f(x)的形式，其中a是下限(min)，b是上限(max)。

如果想具体看一下大概是怎样的，我们设置生成10个数，就会得到下图中的结果

![image-20200721235736763](F:\tf2\image-20200721235736763.png)

### 进行搜索

```python
from scipy.stats import reciprocal
# f(x) = 1/(x*log(b/a)) a <= x <= b

param_distribution = {
    "hidden_layers":[1, 2, 3, 4],
    "layer_size": np.arange(1, 100),
    "learning_rate": reciprocal(1e-4, 1e-2),
}

from sklearn.model_selection import RandomizedSearchCV

random_search_cv = RandomizedSearchCV(sklearn_model,
                                      param_distribution,
                                      n_iter = 10,
                                      cv = 3,
                                      n_jobs = 1)
random_search_cv.fit(x_train_scaled, y_train, epochs = 100,
                     validation_data = (x_valid_scaled, y_valid),
                     callbacks = callbacks)

# cross_validation: 训练集分成n份，n-1训练，最后一份验证.
```

接下来，我们调用`RandomizedSearchCV`()方法，即定义了用于超参数搜索的model，再调用`fit()`函数，即可开始进行超参数的搜索。

[sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

```python
# 官网中提供的方法声明
class sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, return_train_score=False)
```



该方法需要我们提供一个build_model方法，超参数列表，这两个是必要的。

`n_iter`表示了超参数的采样数目，需要我们在运行时间以及模型质量之间做一个权衡，该值越高，采样越全面，随之而来的测试数量也会增加，测试时间也会增加；该值过小，则并不是能很好起到在多种超参数组合中选择一个好的组合的效果。

`n_jobs`代表了并行运行的测试的数量；与处理器数有关，默认是1。设置为-1则代表用上所有的处理器。



`cv`:该搜索过程采用了cross_validation策略，将训练集分成n份，n-1份训练，最后一份验证。cv参数则代表了这个n的取值，cv越大(n越大)，每个超参数组合训练时训练集($\frac{n-1}{n}$)越多，验证集($\frac {1}{n}$)越小。默认为3-5之间的整数(与数据量大小有关)。显然n越大，训练时间会增加，验证集上结果可能会更好(但也可能出现过拟合的情况).

它会先采用cross_validation机制选择超参数，而后会用所有的训练集来训练一次最好的参数。

训练结束后，我们可以通过 

`best_params_`获取最好的超参数

`best_score_`获取最好的得分

`best_estimator_`获取最好的模型

```python
print(random_search_cv.best_params_)
print(random_search_cv.best_score_)
print(random_search_cv.best_estimator_)
```

![image-20200722000526847](F:\tf2\image-20200722000526847.png)

`best_estimator_.model`通过该属性即可获取真正的model，这个model已经是tf的model而不是sklearn的model

```python
model = random_search_cv.best_estimator_.model
model.evaluate(x_test_scaled, y_test)
```

![image-20200722000600040](F:\tf2\image-20200722000600040.png)



## 具体代码

```python
#!pip install sklearn --user
```


```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
```


```python
train = pd.read_csv('../input/train_set.csv', sep='\t')
test = pd.read_csv('../input/test_a.csv', sep='\t')
```


```python
train_text = train['text']
test_text = test['text']
all_text = pd.concat([train_text, test_text])
```

### TF-IDF


```python
%%time
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
train_word_features
```

    CPU times: user 4min 51s, sys: 2.1 s, total: 4min 53s
    Wall time: 4min 53s



```python
X_train = train_word_features
y_train = train['label']

# 可以改变输入维度
x_train_, x_valid_, y_train_, y_valid_ = train_test_split(X_train, y_train, test_size=0.2)
X_test = test_word_features
```

### 简单的逻辑回归


```python
%%time
clf = LogisticRegression(C=4, n_jobs=16)
clf.fit(x_train_, y_train_)

y_pred = clf.predict(x_valid_)
train_scores = clf.score(x_train_, y_train_)
print(train_scores, f1_score(y_pred, y_valid_, average='macro'))
```

    0.94820625 0.9177543223181867
    CPU times: user 1.45 s, sys: 607 ms, total: 2.06 s
    Wall time: 3min 26s

### XGB

```python
class XGB():
    
    def __init__(self, X_df, y_df):
        self.X = X_df
        self.y = y_df
       
    def train(self, param):
        self.model = XGBClassifier(**param)
        self.model.fit(self.X, self.y, eval_set=[(self.X, self.y)],
                       eval_metric=['mlogloss'],
                       early_stopping_rounds=10,  # 连续N次分值不再优化则提前停止
                       verbose=False
                      )
        
#         mode evaluation
        train_result, train_proba = self.model.predict(self.X), self.model.predict_proba(self.X)
        train_acc = accuracy_score(self.y, train_result)
        #train_auc = f1_score(self.y, train_proba, average='macro')
        
        #print("Train acc: %.2f%% Train auc: %.2f" % (train_acc*100.0, train_auc))
        
    def test(self, X_test, y_test):
        result, proba = self.model.predict(X_test), self.model.predict_proba(X_test)
        acc = accuracy_score(y_test, result)
        f1 = f1_score(y_test, proba, average='macro')
        
        print("acc: %.2f%% F1_score: %.2f%%" % (acc*100.0, f1))
    
    def grid(self, param_grid):
        self.param_grid = param_grid
        xgb_model = XGBClassifier(nthread=20)
        clf = GridSearchCV(xgb_model, self.param_grid, scoring='f1_macro', cv=2, verbose=1)
        clf.fit(self.X, self.y)
        print("Best score: %f using parms: %s" % (clf.best_score_, clf.best_params_))
        return clf.best_params_, clf.best_score_

    
```


```python
x_train_, x_valid_, y_train_, y_valid_ = train_test_split(X_train[:, :300], y_train, test_size=0.2, shuffle=True, random_state=42)
X_test = test_word_features[:,:300]
```


```python
%%time
param = {'learning_rate': 0.05,         #  (xgb’s “eta”)
              'objective': 'multi:softmax', 
              'n_jobs': 16,
              'n_estimators': 300,           # 树的个数
              'max_depth': 10,               
              'gamma': 0.5,                  # 惩罚项中叶子结点个数前的参数，Increasing this value will make model more conservative.
              'reg_alpha': 0,               # L1 regularization term on weights.Increasing this value will make model more conservative.
              'reg_lambda': 2,              # L2 regularization term on weights.Increasing this value will make model more conservative.
              'min_child_weight' : 1,      # 叶子节点最小权重
              'subsample':0.8,             # 随机选择80%样本建立决策树
              'random_state':1           # 随机数
             }
model = XGB(x_train_, y_train_)

```

    CPU times: user 8 µs, sys: 0 ns, total: 8 µs
    Wall time: 11.9 µs

XGB即使放在天池实验室提供的GPU环境下，超参数的搜索也都进行了快两个小时。

```python
%%time
model.train(param)
```

    CPU times: user 1h 3min 44s, sys: 1min 58s, total: 1h 5min 43s
    Wall time: 11min 15s



```python
%%time
model.test(x_valid_, y_valid_)
```


```python
final_model = XGB(X_train, y_train)
final_model.train(param)

submission = pd.read_csv('../input/test_a_sample_submit.csv')
preds = final_model.model.predict(X_test)
submission['label'] = preds
submission.to_csv('./xgb_submission.csv', index=False)
```



