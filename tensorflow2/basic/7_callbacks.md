# callbacks

## tf.keras.callbacks.EarlyStopping
```
    Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.

    Args:
        monitor: Quantity to be monitored. Defaults to `"val_loss"`.
        min_delta: Minimum change in the monitored quantity to qualify as an
            improvement, i.e. an absolute change of less than min_delta, will
            count as no improvement. Defaults to `0`.
        patience: Number of epochs with no improvement after which training will
            be stopped. Defaults to `0`.
        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays
            messages when the callback takes an action. Defaults to `0`.
        mode: One of `{"auto", "min", "max"}`. In `min` mode, training will stop
            when the quantity monitored has stopped decreasing; in `"max"` mode
            it will stop when the quantity monitored has stopped increasing; in
            `"auto"` mode, the direction is automatically inferred from the name
            of the monitored quantity. Defaults to `"auto"`.
        baseline: Baseline value for the monitored quantity. If not `None`,
            training will stop if the model doesn't show improvement over the
            baseline. Defaults to `None`.
        restore_best_weights: Whether to restore model weights from the epoch
            with the best value of the monitored quantity. If `False`, the model
            weights obtained at the last step of training are used. An epoch
            will be restored regardless of the performance relative to the
            `baseline`. If no epoch improves on `baseline`, training will run
            for `patience` epochs and restore weights from the best epoch in
            that set. Defaults to `False`.
        start_from_epoch: Number of epochs to wait before starting to monitor
            improvement. This allows for a warm-up period in which no
            improvement is expected and thus training will not be stopped.
            Defaults to `0`.


    Example:

    >>> callback = keras.callbacks.EarlyStopping(monitor='loss',
    ...                                               patience=3)
    >>> # This callback will stop the training when there is no improvement in
    >>> # the loss for three consecutive epochs.
    >>> model = keras.models.Sequential([keras.layers.Dense(10)])
    >>> model.compile(keras.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> len(history.history['loss'])  # Only 4 epochs are run.
    4
```

## tf.keras.callbacks.LearningRateScheduler
```
    Learning rate scheduler.

    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current learning rate, and applies the updated learning rate on
    the optimizer.

    Args:
        schedule: A function that takes an epoch index (integer, indexed from 0)
            and current learning rate (float) as inputs and returns a new
            learning rate as output (float).
        verbose: Integer. 0: quiet, 1: log update messages.

    Example:

    >>> # This function keeps the initial learning rate for the first ten epochs
    >>> # and decreases it exponentially after that.
    >>> def scheduler(epoch, lr):
    ...     if epoch < 10:
    ...         return lr
    ...     else:
    ...         return lr * ops.exp(-0.1)
    >>>
    >>> model = keras.models.Sequential([keras.layers.Dense(10)])
    >>> model.compile(keras.optimizers.SGD(), loss='mse')
    >>> round(model.optimizer.learning_rate, 5)
    0.01

    >>> callback = keras.callbacks.LearningRateScheduler(scheduler)
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=15, callbacks=[callback], verbose=0)
    >>> round(model.optimizer.learning_rate, 5)
    0.00607
```
## tf.keras.callbacks.ModelCheckpoint
```
    Args:
        filepath: string or `PathLike`, path to save the model file.
            `filepath` can contain named formatting options,
            which will be filled the value of `epoch` and keys in `logs`
            (passed in `on_epoch_end`).
            The `filepath` name needs to end with `".weights.h5"` when
            `save_weights_only=True` or should end with `".keras"` when
            checkpoint saving the whole model (default).
            For example:
            if `filepath` is `"{epoch:02d}-{val_loss:.2f}.keras"`, then the
            model checkpoints will be saved with the epoch number and the
            validation loss in the filename. The directory of the filepath
            should not be reused by any other callbacks to avoid conflicts.
        monitor: The metric name to monitor. Typically the metrics are set by
            the `Model.compile` method. Note:
            * Prefix the name with `"val_"` to monitor validation metrics.
            * Use `"loss"` or `"val_loss"` to monitor the model's total loss.
            * If you specify metrics as strings, like `"accuracy"`, pass the
                same string (with or without the `"val_"` prefix).
            * If you pass `metrics.Metric` objects, `monitor` should be set to
                `metric.name`
            * If you're not sure about the metric names you can check the
                contents of the `history.history` dictionary returned by
                `history = model.fit()`
            * Multi-output models set additional prefixes on the metric names.
        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
            displays messages when the callback takes an action.
        save_best_only: if `save_best_only=True`, it only saves when the model
            is considered the "best" and the latest best model according to the
            quantity monitored will not be overwritten. If `filepath` doesn't
            contain formatting options like `{epoch}` then `filepath` will be
            overwritten by each new better model.
        mode: one of {`"auto"`, `"min"`, `"max"`}. If `save_best_only=True`, the
            decision to overwrite the current save file is made based on either
            the maximization or the minimization of the monitored quantity.
            For `val_acc`, this should be `"max"`, for `val_loss` this should be
            `"min"`, etc. In `"auto"` mode, the mode is set to `"max"` if the
            quantities monitored are `"acc"` or start with `"fmeasure"` and are
            set to `"min"` for the rest of the quantities.
        save_weights_only: if `True`, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model is
            saved (`model.save(filepath)`).
        save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
            saves the model after each epoch. When using integer, the callback
            saves the model at end of this many batches. If the `Model` is
            compiled with `steps_per_execution=N`, then the saving criteria will
            be checked every Nth batch. Note that if the saving isn't aligned to
            epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset
            every epoch). Defaults to `"epoch"`.
        initial_value_threshold: Floating point initial "best" value of the
            metric to be monitored. Only applies if `save_best_value=True`. Only
            overwrites the model weights already saved if the performance of
            current model is better than this value.
```


## tf.keras.callbacks.TensorBoard
```
    Args:
        log_dir: the path of the directory where to save the log files to be
            parsed by TensorBoard. e.g.,
            `log_dir = os.path.join(working_dir, 'logs')`.
            This directory should not be reused by any other callbacks.
        histogram_freq: frequency (in epochs) at which to compute
            weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph:  (Not supported at this time)
            Whether to visualize the graph in TensorBoard.
            Note that the log file can become quite large
            when `write_graph` is set to `True`.
        write_images: whether to write model weights to visualize as image in
            TensorBoard.
        write_steps_per_second: whether to log the training steps per second
            into TensorBoard. This supports both epoch and batch frequency
            logging.
        update_freq: `"batch"` or `"epoch"` or integer. When using `"epoch"`,
            writes the losses and metrics to TensorBoard after every epoch.
            If using an integer, let's say `1000`, all metrics and losses
            (including custom ones added by `Model.compile`) will be logged to
            TensorBoard every 1000 batches. `"batch"` is a synonym for 1,
            meaning that they will be written every batch.
            Note however that writing too frequently to TensorBoard can slow
            down your training, especially when used with distribution
            strategies as it will incur additional synchronization overhead.
            Batch-level summary writing is also available via `train_step`
            override. Please see
            [TensorBoard Scalars tutorial](
                https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)  # noqa: E501
            for more details.
        profile_batch: (Not supported at this time)
            Profile the batch(es) to sample compute characteristics.
            profile_batch must be a non-negative integer or a tuple of integers.
            A pair of positive integers signify a range of batches to profile.
            By default, profiling is disabled.
        embeddings_freq: frequency (in epochs) at which embedding layers will be
            visualized. If set to 0, embeddings won't be visualized.
        embeddings_metadata: Dictionary which maps embedding layer names to the
            filename of a file in which to save metadata for the embedding layer.
            In case the same metadata file is to be
            used for all embedding layers, a single filename can be passed.
```

## 