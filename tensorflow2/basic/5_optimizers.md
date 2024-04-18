# optimizers
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

# adam


