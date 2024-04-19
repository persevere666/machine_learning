# metrics
```python
ALL_OBJECTS = {
    # Base
    Metric,
    Mean,
    Sum,
    MeanMetricWrapper,
    # Regression
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredLogarithmicError,
    CosineSimilarity,
    LogCoshError,
    R2Score,
    # Classification
    AUC,
    FalseNegatives,
    FalsePositives,
    Precision,
    PrecisionAtRecall,
    Recall,
    RecallAtPrecision,
    SensitivityAtSpecificity,
    SpecificityAtSensitivity,
    TrueNegatives,
    TruePositives,
    # Hinge
    Hinge,
    SquaredHinge,
    CategoricalHinge,
    # Probabilistic
    KLDivergence,
    Poisson,
    BinaryCrossentropy,
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
    # Accuracy
    Accuracy,
    BinaryAccuracy,
    CategoricalAccuracy,
    SparseCategoricalAccuracy,
    TopKCategoricalAccuracy,
    SparseTopKCategoricalAccuracy,
    # F-Score
    F1Score,
    FBetaScore,
    # IoU
    IoU,
    BinaryIoU,
    MeanIoU,
    OneHotIoU,
    OneHotMeanIoU,
}
ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)

ALL_OBJECTS_DICT.update(
    {
        "bce": BinaryCrossentropy,
        "BCE": BinaryCrossentropy,
        "mse": MeanSquaredError,
        "MSE": MeanSquaredError,
        "mae": MeanAbsoluteError,
        "MAE": MeanAbsoluteError,
        "mape": MeanAbsolutePercentageError,
        "MAPE": MeanAbsolutePercentageError,
        "msle": MeanSquaredLogarithmicError,
        "MSLE": MeanSquaredLogarithmicError,
    }
)
```
## 1 Base
### 1.1 Mean

### 1.2 Sum


## 2 Regression
### 2.1 MeanSquaredError
套壳 tf.keras.losses.mean_squared_error
### 2.2 RootMeanSquaredError
套壳 tf.keras.losses.mean_squared_error
### 2.3 MeanAbsoluteError
套壳 tf.keras.losses.mean_absolute_error
### 2.4 MeanAbsolutePercentageError
套壳 tf.keras.losses.mean_absolute_percentage_error
```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true))
```
### 2.5 MeanSquaredLogarithmicError
套壳 tf.keras.losses.mean_squared_logarithmic_error
```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
```

### 2.6 CosineSimilarity
调用 cosine_similarity
```python
    y_pred = normalize(y_pred, axis=axis)
    y_true = normalize(y_true, axis=axis)
    return ops.sum(y_true * y_pred, axis=axis)
```
### 2.7 LogCoshError
套壳 tf.keras.losses.log_cosh
```python
    error = y_pred - y_true
    logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)
```
### 2.8 R2Score
```python
    sum_squares_residuals = sum((y_true - y_pred) ** 2)
    sum_squares = sum((y_true - mean(y_true)) ** 2)
    R2 = 1 - sum_squares_residuals / sum_squares
```

## 3 Classification
各个指标的定义
准确率/正确率(accuracy) = 所有预测正确的样本数量 /总样本数量，即 (TP+TN) / (TP+TN+FP+FN)。

精确率(precision) = 预测正确的正类样本数量 / 预测为正类的样本数量 TP/(TP+FP)，即预测为正的样本中，真正的正类占比。

召回率(recall) = 预测正确的正类样本数量 / 实际为正类的样本数量 TP/(TP+FN)，即所有正类中，正确预测的占比。

F1值 = 精确率 * 召回率 * 2 / (精确率 + 召回率) （F 值即为精确率和召回率的调和均值）

真阳性率(TPR) = TP/(TP+FN)，即所有正类样本中被正确分为正类的比例，计算方式和召回率相同。

假阳性率(FPR) = FP/(FP+TN)，即所有负类样本中被错误分为正类的比例。

随着预测为正类的阈值变化，TPR和FPR相应地变化，因此可以得到以TPR为纵坐标和FPR为横坐标的曲线，即ROC曲线，因此可以得到AUC。

