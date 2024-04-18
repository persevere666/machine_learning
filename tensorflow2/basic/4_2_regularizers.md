# regularizers
```python
ALL_OBJECTS = {
    Regularizer,
    L1,
    L2,
    L1L2,
    OrthogonalRegularizer,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)
```

##
