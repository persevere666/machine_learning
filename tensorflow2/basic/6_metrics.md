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

### 3.1 AUC
太复杂，暂时不看

### 3.2 FalseNegatives
```python
m = tf.keras.metrics.FalseNegatives()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
m.result()
#2.0

m.reset_state()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0], sample_weight=[0, 0, 1, 0])
m.result()
#1.0
```
### 3.3 FalsePositives
```python
m = tf.keras.metrics.FalsePositives()
m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
m.result()
#2.0

m.reset_state()
m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
m.result()
#1.0
```

### 3.4 PrecisionAtRecall
Computes best precision where recall is >= specified value
```python
m = keras.metrics.PrecisionAtRecall(0.5)
m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
m.result()
#0.5

m.reset_state()
m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8], sample_weight=[2, 2, 2, 1, 1])
m.result()
#0.33333333

#Usage with `compile()` API:
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=[keras.metrics.PrecisionAtRecall(recall=0.8)]
)
```

### 3.5 Recall
    Computes the recall of the predictions with respect to the labels.

    This metric creates two local variables, `true_positives` and
    `false_negatives`, that are used to compute the recall. This value is
    ultimately returned as `recall`, an idempotent operation that simply divides
    `true_positives` by the sum of `true_positives` and `false_negatives`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `top_k` is set, recall will be computed as how often on average a class
    among the labels of a batch entry is in the top-k predictions.

    If `class_id` is specified, we calculate recall by considering only the
    entries in the batch for which `class_id` is in the label, and computing the
    fraction of them for which `class_id` is above the threshold and/or in the
    top-k predictions.

    Args:
        thresholds: (Optional) A float value, or a Python list/tuple of float
            threshold values in `[0, 1]`. A threshold is compared with
            prediction values to determine the truth value of predictions (i.e.,
            above the threshold is `True`, below is `False`). If used with a
            loss function that sets `from_logits=True` (i.e. no sigmoid
            applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
            If neither `thresholds` nor `top_k` are set,
            the default is to calculate recall with `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating recall.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.Recall()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    0.6666667

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
                loss='binary_crossentropy',
                metrics=[keras.metrics.Recall()])
```

Usage with a loss with `from_logits=True`:

```python
model.compile(optimizer='adam',
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.Recall(thresholds=0)])
```


### 3.6 RecallAtPrecision
Computes best recall where precision is >= specified value.

    For a given score-label-distribution the required precision might not
    be achievable, in this case 0.0 is returned as recall.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the recall at the given precision. The threshold for the given
    precision value is computed and used to evaluate the corresponding recall.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    Args:
        precision: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds
            to use for matching the given precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.RecallAtPrecision(0.8)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result()
    1.0

Usage with `compile()` API:
```python
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=[keras.metrics.RecallAtPrecision(precision=0.8)])
```


### 3.7 SensitivityAtSpecificity
### 3.8 SpecificityAtSensitivity
### 3.9 TrueNegatives
    Calculates the number of true negatives.

    If `sample_weight` is given, calculates the sum of the weights of
    true negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of true negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.TrueNegatives()
    >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

### 3.10 TruePositives
    Calculates the number of true positives.

    If `sample_weight` is given, calculates the sum of the weights of
    true positives. This metric creates one local variable, `true_positives`
    that is used to keep track of the number of true positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.TruePositives()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0
