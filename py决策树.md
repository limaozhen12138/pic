1. cross_val_score交叉验证评估

```python
estimator:估计方法对象(分类器) `

`X：数据特征(Features)`

` y：数据标签(Labels) ` 

`cv：几折交叉验证`
```

2.

sklrean构建的CART决策树模型官方文档  [✔](https://scikit-learn.org/stable/modules/tree.html#tree)

3.GridSearchCV

[官方链接](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)

GridSearchCV的名字其实可以拆分为两部分，GridSearch和CV，即网格搜索和交叉验证。网格搜索，搜索的是参数，即在指定的参数范围内，按步长依次调整参数，利用调整的参数训练学习器，从所有的参数中找到在验证集上精度最高的参数，这其实是一个训练和比较的过程。k折交叉验证将所有数据集分成k份，不重复地每次取其中一份做测试集，用其余k-1份做训练集训练模型，之后计算该模型在测试集上的得分,将k次的得分取平均得到最后的得分。

参数说明

```python
grid = GridSearchCV(
    dtModel, paramGrid, cv=10,
    return_train_score=True
)
# 这里评估器是我们建立的决策树模型，
# param_grid参数是需要最优化的参数的取值所建立的字典和列表，上面我们设置过
# cv=10是指分成10份，进行交叉验证
# return_train_score 返回计算训练集分数，训练集分数用于了解不同的参数设置如何影响过拟合或欠拟合的权衡。但是，在训练集上计算分数可能在计算上很耗时，并且不严格要求选择产生最优泛化性能的参数。
*************************************
修正后为什么要使用fit，fit有隐式传参
sampled_params = ParameterSampler(self.param_distributions,
                                          self.n_iter,
                                          random_state=self.random_state)
        return self._fit(X, y, groups, sampled_params)
grid.fit()  ：运行网格搜索




****************************************************
GridSearchCV属性说明
　　　　（1） cv_results_ : dict of numpy (masked) ndarrays

　　　　具有键作为列标题和值作为列的dict，可以导入到DataFrame中。注意，“params”键用于存储所有参数候选项的参数设置列表。

　　　　（2） best_estimator_ : estimator

　　　　通过搜索选择的估计器，即在左侧数据上给出最高分数（或指定的最小损失）的估计器。如果refit = False，则不可用。

　　　　（3）best_score_ ：float  best_estimator的分数

　　　　（4）best_parmas_ : dict  在保存数据上给出最佳结果的参数设置

　　　　（5） best_index_ : int 对应于最佳候选参数设置的索引（cv_results_数组）

　　　　search.cv_results _ ['params'] [search.best_index_]中的dict给出了最佳模型的参数设置，给出了最高的平均分数（search.best_score_）。

　　　　（6）scorer_ : function

　　　　Scorer function used on the held out data to choose the best parameters for the model.

　　　　（7）n_splits_ : int

　　　　The number of cross-validation splits (folds/iterations).
```

3.OneHot

[ onehot的使用条件 ]( https://www.cnblogs.com/zongfa/p/9305657.html)

[onehot 编码器官方](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

```pthon
drop='first'

# 指定用于删除每个要素的一个类别的方法。这在完全共线特征导致问题的情况下非常有用，例如将结果数据馈送到神经网络或未规范回归时。

# 但是，删除一个类别会破坏原始表示的对称性，因此可能会在下游模型中引起偏差，例如对于惩罚线性分类或回归模型。
```

4. hstack()函数 
  函数原型：hstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组。

  ```python
  a=[1,2,3]
  b=[4,5,6]
  print(np.hstack((a,b)))
  
  结果：
  [1 2 3 4 5 6 ]
  
  
  import numpy as np
  a=[[1],[2],[3]]
  b=[[1],[2],[3]]
  c=[[1],[2],[3]]
  d=[[1],[2],[3]]
  print(np.hstack((a,b,c,d)))
  输出：
  
  [[1 1 1 1]
   [2 2 2 2]
   [3 3 3 3]]
  它其实就是水平(按列顺序)把数组给堆叠起来
  ```

之后是模型处理完后面的结果查看，数据显示和图片保存，意义不大，把上面的几个方法弄明白就够了