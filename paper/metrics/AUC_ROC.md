# AUC
## ROC和AUC的定义
参考：https://zhuanlan.zhihu.com/p/460549028
AUC定义：
    真阳性率(TPR) = TP/(TP+FN)，即所有正类样本中被正确分为正类的比例，计算方式和召回率相同。
    假阳性率(FPR) = FP/(FP+TN)，即所有负类样本中被错误分为正类的比例。
    随着预测为正类的阈值变化，TPR和FPR相应地变化，因此可以得到以TPR为纵坐标和FPR为横坐标的曲线，即ROC曲线，因此可以得到AUC

## AUC的计算方式
参考：https://blog.csdn.net/pearl8899/article/details/126129148
### 方式一
在有M个正样本,N个负样本的数据集里。一共有M*N对样本（一对样本即，一个正样本与一个负样本）。统计这M*N对样本里，正样本的预测概率大于负样本的预测概率的个数。
$$ AUC = \frac{\sum{I(P_{pos},P_{neg})}}{M*N}$$
$$
I(P_{pos},P_{neg}) = \begin{cases}
    1 & P_{pos} > P_{neg} \\
    0.5 & P_{pos} = P_{neg} \\
    0  & P_{pos} < P_{neg}
\end{cases}
$$
### 方式二
$$
AUC = \frac{\sum_{s_i \in {positiveclass}}rank_{s_i}-\frac{M*(M+1)}{2}}{M*N}
$$
注意：遇到相等得分时，将相等得分的rank取平均值
### 计算代码
```python
import numpy as np
from sklearn.metrics import roc_auc_score

# python sklearn包计算auc
def get_auc(y_labels, y_scores):
    auc = roc_auc_score(y_labels, y_scores)
    print('AUC calculated by sklearn tool is {}'.format(auc))
    return auc

# 方法1计算auc
def calculate_auc_func1(y_labels, y_scores):
    pos_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 1]
    neg_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 0]

    sum_indicator_value = 0
    for i in pos_sample_ids:
        for j in neg_sample_ids:
            if y_scores[i] > y_scores[j]:
                sum_indicator_value += 1
            elif y_scores[i] == y_scores[j]:
                sum_indicator_value += 0.5

    auc = sum_indicator_value/(len(pos_sample_ids) * len(neg_sample_ids))
    print('AUC calculated by function1 is {:.2f}'.format(auc))
    return auc

# 方法2计算auc, 当预测分相同时，未按照定义使用排序值的均值，而是直接使用排序值，当数据量大时，对auc影响小
def calculate_auc_func2(y_labels, y_scores):
    samples = list(zip(y_scores, y_labels))
    rank = [(values2, values1) for values1, values2 in sorted(samples, key=lambda x:x[0])]
    pos_rank = [i+1 for i in range(len(rank)) if rank[i][0] == 1]
    pos_cnt = np.sum(y_labels == 1)
    neg_cnt = np.sum(y_labels == 0)
    auc = (np.sum(pos_rank) - pos_cnt*(pos_cnt+1)/2) / (pos_cnt*neg_cnt)
    print('AUC calculated by function2 is {:.2f}'.format(auc))
    return auc


if __name__ == '__main__':
    y_labels = np.array([1, 1, 0, 0, 0])
    y_scores = np.array([0.4, 0.8, 0.2, 0.4, 0.5])
    get_auc(y_labels, y_scores)
    calculate_auc_func1(y_labels, y_scores)
    calculate_auc_func2(y_labels, y_scores)
```

## AUC的优势
单个指标对模型评估存在多种问题，从而无法真实地评估模型的性能，因此通常综合考虑多个指标。比如F值，综合考虑精确率和召回率，比如AUC，综合考虑TPR和FPR。

AUC作为指标衡量模型时，不依赖于分类阈值的选取，而准确率、精确率、召回率、F1值对阈值的选取依赖大，不同的阈值会带来不同的结果，而从AUC的定义(ROC的生成)知道，AUC是根据所有分类阈值得到的，因此比单独依赖一个分类阈值的指标更有优势。AUC体现的是对样本的排序能力，与具体的分值无关，和推荐系统中的大多数业务场景更贴合，因为大多数业务场景关心item之间的相对序而不关心item的预测分。

AUC对正负样本比例不敏感，也是它在业界被广泛使用的原因。
### AUC对正负样本比例不敏感的原因
参考：https://www.modb.pro/db/176229


## 多分类情况下

参考：https://blog.csdn.net/PrimiHub/article/details/134161420
现在考虑多分类的情况，假设类别数为C CC。

一种想法是将某一类别设为正样本类别，其余类别设为负样本类别，然后计算二分类下的 AUC。这种方法叫做一对多，即 One-Vs-Rest (OVR)。可以得到C CC个二分类的 AUC，然后计算平均数得到多分类的 AUC。

另一种想法是将某一类别设为正样本类别，另外一个类别（非自身）设为负样本类别计算二分类的 AUC。这种方法叫做一对一，即 One-Vs-One (OVO)。可以得到C ( C − 1 ) C(C-1)C(C−1)个二分类的 AUC，然后计算平均数。

当计算平均数时，可以考虑算数平均数（称为macro），或者加权平均数（称为weighted）。其中，加权为各类别的样本所占比例。因此，两两组合可以的得到四种计算多分类 AUC 的方法。值得一提的是，知名机器学习库 scikit-learn 的 roc_auc_score 函数 包含了上述四种方法。

1. 一对多 + 算数平均数（OVR + macro）
2. 一对多 + 加权平均数（OVR + weighted）
3. 一对一 + 算数平均数（OVO + macro）
4. 一对一 + 加权平均数（OVO + weighted）
```python
sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
```

### 二分类下计算AUC
```python
def auc_from_roc(fpr, tpr):
    """
    计算ROC面积
    fpr: 从小到大排序的fpr坐标
    tpr: 从小到大排序的tpr坐标
    """
    area = 0
    for i in range(len(fpr) - 1):
        area += trapezoid_area(fpr[i], fpr[i + 1], tpr[i], tpr[i + 1])
    return area
    
def trapezoid_area(x1, x2, y1, y2):
    """
    计算梯形面积
    x1, x2: 横坐标 (x1 <= x2)
    y1, y2: 纵坐标 (y1 <= y2)
    """
    base = x2 - x1
    height_avg = (y1 + y2) / 2
    return base * height_avg
import numpy as np
# 参考论文
def auc_binary(y_true, y_score, pos_label):
    """
    y_true：真实标签
    y_score：模型预测分数
    pos_label：正样本标签，如“1”
    """
    num_positive_examples = (y_true == pos_label).sum()
    num_negtive_examples = len(y_true) - num_positive_examples

    tp, fp, tp_prev, fp_prev, area = 0, 0, 0, 0, 0
    score = -np.inf

    for i in np.flip(np.argsort(y_score)):
        if y_score[i] != score:
            area += trapezoid_area(fp_prev, fp, tp_prev, tp)
            score = y_score[i]
            fp_prev = fp
            tp_prev = tp

        if y_true[i] == pos_label:
            tp += 1
        else:
            fp += 1

    area += trapezoid_area(fp_prev, fp, tp_prev, tp)
    area /= num_positive_examples * num_negtive_examples

    return area

```

### 一对多 + 算数平均数
```python
# sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
def auc_ovr_macro(y_true, y_score):
    auc = 0
    C = max(y_true) + 1

    for i in range(C):
        auc += auc_binary(y_true, y_score[:, i], pos_label=i)

    return auc / C
```
### 一对多 + 加权平均数
```python
# sklearn.metrics.roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
def auc_ovr_weighted(y_true, y_score):
    auc = 0
    C = max(y_true) + 1
    n = len(y_true)

    for i in range(C):
        p = sum(y_true == i) / n
        auc += auc_binary(y_true, y_score[:, i], pos_label=i) * p

    return auc
```
### 一对一 + 算数平均数
```python
# sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
def auc_ovo_macro(y_true, y_score):
    auc = 0
    C = max(y_true) + 1

    for i in range(C - 1):
        i_index = np.where(y_true == i)[0]
        for j in range(i + 1, C):
            j_index = np.where(y_true == j)[0]
            index = np.concatenate((i_index, j_index))

            auc_i_j = auc_binary(y_true[index], y_score[index, i], pos_label=i)
            auc_j_i = auc_binary(y_true[index], y_score[index, j], pos_label=j)
            auc += (auc_i_j + auc_j_i) / 2

    return auc * 2 / (C * (C - 1))

```

### 
```python
# sklearn.metrics.roc_auc_score(y_true, y_score, average='weighted', multi_class='ovo')
def auc_ovo_weighted(y_true, y_score):
    auc = 0
    C = max(y_true) + 1
    n = len(y_true)

    for i in range(C - 1):
        i_index = np.where(y_true == i)[0]
        for j in range(i + 1, C):
            j_index = np.where(y_true == j)[0]
            index = np.concatenate((i_index, j_index))

            p = len(index) / n / (C - 1)
            auc_i_j = auc_binary(y_true[index], y_score[index, i], pos_label=i)
            auc_j_i = auc_binary(y_true[index], y_score[index, j], pos_label=j)
            auc += (auc_i_j + auc_j_i) / 2 * p

    return auc
```

## 联邦学习中的 ROC 平均
参考：https://blog.csdn.net/PrimiHub/article/details/131846219
### 垂直平均
```python
import numpy as np
def roc_vertical_avg(samples, FPR, TPR):
    """
    samples：选取FPR点的个数
    FPR：包含所有FPR的列表
    TPR：包含所有TPR的列表
    """
    nrocs = len(FPR)
    tpravg = []
    fpr = [i / samples for i in range(samples + 1)]

    for fpr_sample in fpr:
        tprsum = 0
        # 将所有计算的tpr累加
        for i in range(nrocs):
            tprsum += tpr_for_fpr(fpr_sample, FPR[i], TPR[i])
        # 计算平均的tpr
        tpravg.append(tprsum / nrocs)

    return fpr, tpravg

# 计算对应fpr的tpr
def tpr_for_fpr(fpr_sample, fpr, tpr):
    i = 0
    while i < len(fpr) - 1 and fpr[i + 1] <= fpr_sample:
        i += 1

    if fpr[i] == fpr_sample:
        return tpr[i]
    else:
        return interpolate(fpr[i], tpr[i], fpr[i + 1], tpr[i + 1], fpr_sample)

# 插值
def interpolate(fprp1, tprp1, fprp2, tprp2, x):
    slope = (tprp2 - tprp1) / (fprp2 - fprp1)
    return tprp1 + slope * (x - fprp1)

```
### 阈值平均
```python
import numpy as np
def roc_threshold_avg(samples, FPR, TPR, THRESHOLDS):
    """
    samples：选取FPR点的个数
    FPR：包含所有FPR的列表
    TPR：包含所有TPR的列表
    THRESHOLDS：包含所有THRESHOLDS的列表
    """
    nrocs = len(FPR)
    T = []
    fpravg = []
    tpravg = []

    for thresholds in THRESHOLDS:
        for t in thresholds:
            T.append(t)
    T.sort(reverse=True)

    for tidx in range(0, len(T), int(len(T) / samples)):
        fprsum = 0
        tprsum = 0
        # 将所有计算的fpr和tpr累加
        for i in range(nrocs):
            fprp, tprp = roc_point_at_threshold(FPR[i], TPR[i], THRESHOLDS[i], T[tidx])
            fprsum += fprp
            tprsum += tprp
        # 计算平均的fpr和tpr
        fpravg.append(fprsum / nrocs)
        tpravg.append(tprsum / nrocs)

    return fpravg, tpravg

# 计算对应threshold的fpr和tpr
def roc_point_at_threshold(fpr, tpr, thresholds, thresh):
    i = 0
    while i < len(fpr) - 1 and thresholds[i] > thresh:
        i += 1
    return fpr[i], tpr[i]

```