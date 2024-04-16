# tf中的__init__和build方法和call方法

在使用tf构建网络框架的时候，经常会遇到__init__、build 和call这三个互相搭配着使用，那么它们的区别主要在哪里呢？

1）__init__主要用来做参数初始化用，比如我们要初始化卷积的一些参数，就可以放到这里面，这个函数用于对所有独立的输入进行初始化。（独立的输入：特指和训练数据无关的输入）(这个函数仅被执行一次)

2）call可以把类型的对象当做函数来使用，这个对象可以是在__init__里面也可以是在build里面

3）build一般是和call搭配使用，这个时候，它的功能和__init__很相似，当build中存放本层需要初始化的变量，当call被第一次调用的时候，会先执行build()方法初始化变量，但后面再调用到call的时候，是不会再去执行build()方法初始化变量，(注意：这个函数仅在Call被第一次调用时执行)


# layers
## tf.keras.layers.Conv1D
reference: https://blog.csdn.net/yinizhilianlove/article/details/127129520

# RNN
* Inherits From: Layer, Operation
```
tf.keras.layers.RNN(
    cell,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    zero_output_for_mask=False,
    **kwargs
)
```
# optimizer种类
```
# keras/optimizers/__init__.py中定义
ALL_OBJECTS = {
    Optimizer,
    Adam,
    SGD,
    RMSprop,
    Adadelta,
    AdamW,
    Adagrad,
    Adamax,
    Adafactor,
    Nadam,
    Ftrl,
    Lion,
    LossScaleOptimizer,
}
ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}
```
# loss种类
```
# keras/losses/__init__.py中定义 
ALL_OBJECTS = {
    # Base
    Loss,
    LossFunctionWrapper,
    # Probabilistic
    KLDivergence,
    Poisson,
    BinaryCrossentropy,
    BinaryFocalCrossentropy,
    CategoricalCrossentropy,
    CategoricalFocalCrossentropy,
    SparseCategoricalCrossentropy,
    # Regression
    MeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredLogarithmicError,
    CosineSimilarity,
    LogCosh,
    Huber,
    # Hinge
    Hinge,
    SquaredHinge,
    CategoricalHinge,
    # Image segmentation
    Dice,
    # Probabilistic
    kl_divergence,
    poisson,
    binary_crossentropy,
    binary_focal_crossentropy,
    categorical_crossentropy,
    categorical_focal_crossentropy,
    sparse_categorical_crossentropy,
    # Regression
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    cosine_similarity,
    log_cosh,
    huber,
    # Hinge
    hinge,
    squared_hinge,
    categorical_hinge,
    # Image segmentation
    dice,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {
        "bce": binary_crossentropy,
        "BCE": binary_crossentropy,
        "kld": kl_divergence,
        "KLD": kl_divergence,
        "mae": mean_absolute_error,
        "MAE": mean_absolute_error,
        "mse": mean_squared_error,
        "MSE": mean_squared_error,
        "mape": mean_absolute_percentage_error,
        "MAPE": mean_absolute_percentage_error,
        "msle": mean_squared_logarithmic_error,
        "MSLE": mean_squared_logarithmic_error,
    }
)
```
## poisson Poisson
```python
    loss = y_pred - y_true * log(y_pred)
```
```
    return ops.mean(y_pred - y_true * ops.log(y_pred + epsilon), axis=-1)
```

## mae MAE mean_absolute_error MeanAbsoluteError

## mape MAPE mean_absolute_percentage_error MeanAbsolutePercentageError
```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)
```
```source code
    diff = ops.abs((y_true - y_pred) / ops.maximum(ops.abs(y_true), epsilon))
    return 100.0 * ops.mean(diff, axis=-1)  
```
## mse MSE mean_squared_error MeanSquaredError
## msle MSLE mean_squared_logarithmic_error MeanSquaredLogarithmicError
```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
```
```
    first_log = ops.log(ops.maximum(y_pred, epsilon) + 1.0)
    second_log = ops.log(ops.maximum(y_true, epsilon) + 1.0)
    return ops.mean(ops.square(first_log - second_log), axis=-1)
```
## kld KLD kl_divergence KLDivergence
```python
    loss = y_true * log(y_true / y_pred)
```
```source code
def kl_divergence(y_true, y_pred):
    ...
    y_true = ops.clip(y_true, backend.epsilon(), 1)
    y_pred = ops.clip(y_pred, backend.epsilon(), 1)
    return ops.sum(y_true * ops.log(y_true / y_pred), axis=-1)
```
* KL散度和交叉熵
  * KL散度 $KL(P|Q)=\sum{y_{true}}*log(\frac{y_{true}}{y_{pred}})$


## bce BCE binary_crossentropy BinaryCrossentropy
```
def binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    ...
    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    return ops.mean(
        ops.binary_crossentropy(y_true, y_pred, from_logits=from_logits),
        axis=axis,
    )
```
```ops.binary_crossentropy
    if any_symbolic_tensors((target, output)):
        return BinaryCrossentropy(from_logits=from_logits).symbolic_call(
            target, output
        )
    return backend.nn.binary_crossentropy(
        target, output, from_logits=from_logits
    )
```

```backend.nn.binary_crossentropy
    if from_logits:
        output = sigmoid(output)

    output = np.clip(output, epsilon(), 1.0 - epsilon())
    bce = target * np.log(output)
    bce += (1.0 - target) * np.log(1.0 - output)
    return -bce
```

## binary_focal_crossentropy BinaryFocalCrossentropy

```
def binary_focal_crossentropy(
    y_true,
    y_pred,
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        y_pred = ops.sigmoid(y_pred)

    bce = ops.binary_crossentropy(
        target=y_true,
        output=y_pred,
        from_logits=False,
    )

    # Calculate focal factor
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_factor = ops.power(1.0 - p_t, gamma)

    focal_bce = focal_factor * bce

    if apply_class_balancing:
        weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_bce = weight * focal_bce

    return ops.mean(focal_bce, axis=axis)
```
## categorical_crossentropy CategoricalCrossentropy
```keras/losses/losses.py
    if label_smoothing:
        num_classes = ops.cast(ops.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )

    return ops.categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, axis=axis
    )
```
```keras/ops/nn.py ops.categorical_crossentropy 
    if any_symbolic_tensors((target, output)):
        return CategoricalCrossentropy(
            from_logits=from_logits, axis=axis
        ).symbolic_call(target, output)
    return backend.nn.categorical_crossentropy(
        target, output, from_logits=from_logits, axis=axis
    )

```
```keras/backend/nn.py backend.nn.categorical_crossentropy
    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / np.sum(output, axis, keepdims=True)
        output = np.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = np.log(output)
    return -np.sum(target * log_prob, axis=axis)
```
```
    def log_softmax(x, axis=None):
        max_x = np.max(x, axis=axis, keepdims=True)
        logsumexp = np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True))
        return x - max_x - logsumexp
```
## categorical_focal_crossentropy CategoricalFocalCrossentropy
```keras/losses/losses.py
def categorical_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    ...
    if label_smoothing:
        num_classes = ops.cast(ops.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )

    if from_logits:
        y_pred = ops.softmax(y_pred, axis=axis)

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = y_pred / ops.sum(y_pred, axis=axis, keepdims=True)
    output = ops.clip(output, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate cross entropy
    cce = -y_true * ops.log(output)

    # Calculate factors
    modulating_factor = ops.power(1.0 - output, gamma)
    weighting_factor = ops.multiply(modulating_factor, alpha)

    # Apply weighting factor
    focal_cce = ops.multiply(weighting_factor, cce)
    focal_cce = ops.sum(focal_cce, axis=axis)
    return focal_cce

```

## sparse_categorical_crossentropy SparseCategoricalCrossentropy
```
    if ignore_class is not None:
        res_shape = ops.shape(y_pred)[:-1]
        valid_mask = ops.not_equal(y_true, ops.cast(ignore_class, y_pred.dtype))
        y_true = y_true * ops.cast(valid_mask, y_true.dtype)
        y_pred = y_pred * ops.cast(
            ops.expand_dims(valid_mask, -1), y_pred.dtype
        )

    res = ops.sparse_categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=from_logits,
        axis=axis,
    )

    if ignore_class is not None:
        valid_mask = ops.reshape(valid_mask, res_shape)
        res = ops.where(valid_mask, res, 0.0)

        try:
            res._keras_mask = valid_mask
        except AttributeError:
            pass

    return res
```

## cosine_similarity CosineSimilarity

