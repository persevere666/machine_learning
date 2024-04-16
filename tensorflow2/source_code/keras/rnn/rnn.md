# keras源码之rnn

## 初始化
```
    def __init__(
        self,
        cell,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        zero_output_for_mask=False,
        **kwargs,
    )
```
## call函数
```
    def call(
        self,
        sequences,
        initial_state=None,
        mask=None,
        training=False,
    )

```
* call内部调用inner_loop
```
    ...
    last_output, outputs, states = self.inner_loop(
        sequences=sequences,
        initial_state=initial_state,
        mask=mask,
        training=training,
    )
    ...
    #返回值
    if self.return_sequences:
        output = outputs
    else:
        output = last_output

    if self.return_state:
        if len(states) == 1:
            state = states[0]
            return output, state
        return output, *states
    return output

```
* inner_loop内部
```
    def inner_loop(self, sequences, initial_state, mask, training=False):
        cell_kwargs = {}
        if isinstance(self.cell, Layer) and self.cell._call_has_training_arg:
            cell_kwargs["training"] = training

        def step(inputs, states):
            output, new_states = self.cell(inputs, states, **cell_kwargs)
            if not tree.is_nested(new_states):
                new_states = [new_states]
            return output, new_states

        if not tree.is_nested(initial_state):
            initial_state = [initial_state]

        return backend.rnn(
            step,
            sequences,
            initial_state,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=sequences.shape[1],
            zero_output_for_mask=self.zero_output_for_mask,
            return_all_outputs=self.return_sequences,
        )

```
## 先看tensorflow版本的rnn(是一个方法)
1. 初始化
```
def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
```
2. 内部逻辑
```
   1. unroll==true
   
   2. unroll==false
        while_loop_kwargs = {
            "cond": lambda time, *_: time < time_steps_t,
            "maximum_iterations": max_iterations,
            "parallel_iterations": 32,
            "swap_memory": True,
        }
        if masking_fn is not None:
            def _step(time, output_ta_t, prev_output, *states):
                """RNN step function.

                    Args:
                        time: Current timestep value.
                        output_ta_t: TensorArray.
                        prev_output: tuple of outputs from time - 1.
                        *states: List of states.

                    Returns:
                        Tuple: `(time + 1, output_ta_t, output) + tuple(new_states)`
                    """
                ...
                output, new_states = step_function(
                        current_input, tuple(states) + tuple(constants)
                )
                ...
                return (time + 1, output_ta_t, tuple(flat_new_output)) + tuple(
                    new_states
                )
            final_outputs = tf.while_loop(
                body=_step,
                loop_vars=(time, output_ta, flat_zero_output) + states,
                **while_loop_kwargs,
            )
            # Skip final_outputs[2] which is the output for final timestep.
            new_states = final_outputs[3:]
        else:
            def _step(time, output_ta_t, *states):
                """RNN step function.
                Args:
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    *states: List of states.

                Returns:
                    Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                """
                ...
                output, new_states = step_function(
                    current_input, tuple(states) + tuple(constants)
                )
                ...
                return (time + 1, output_ta_t) + tuple(new_states)
            final_outputs = tf.while_loop(
                body=_step,
                loop_vars=(time, output_ta) + states,
                **while_loop_kwargs,
            )
            new_states = final_outputs[2:]
```    

### rnn内部的mask