# tf中的__init__和build方法和call方法
在使用tf构建网络框架的时候，经常会遇到__init__、build 和call这三个互相搭配着使用，那么它们的区别主要在哪里呢？

1）__init__主要用来做参数初始化用，比如我们要初始化卷积的一些参数，就可以放到这里面，这个函数用于对所有独立的输入进行初始化。（独立的输入：特指和训练数据无关的输入）(这个函数仅被执行一次)

2）call可以把类型的对象当做函数来使用，这个对象可以是在__init__里面也可以是在build里面

3）build一般是和call搭配使用，这个时候，它的功能和__init__很相似，当build中存放本层需要初始化的变量，当call被第一次调用的时候，会先执行build()方法初始化变量，但后面再调用到call的时候，是不会再去执行build()方法初始化变量，(注意：这个函数仅在Call被第一次调用时执行)


# CNN
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


## LSTM

![LSTM](./image/LSTM.png)

1. 遗忘门
    顾名思义，遗忘门用来控制在元胞(cell)状态里哪些信息需要进行遗忘，以使在$C_t$流动的过程中进行适当的更新。它接收$h_{t-1}$和$x_t$作为输入参数，通过sigmoid层得到对应的遗忘门的参数。具体公式如下
    $$
        f_t=sigmoid(W_f*[h_{t-1},x_t]+b_f)
    $$

2. 输入门
   接下来就需要更新细胞状态${C_t}$了。首先LSTM需要生成一个用来更新的候选值，记为$\tilde{C_t}$，通过tanh层来实现。然后还需要一个输入门参数$i_t$来决定更新的信息，同样通过sigmoid层实现。最后将$i_t$和$\tilde{C_t}$相乘得到更新的信息，同时将上面得到的遗忘门$f_t$和元胞状态$C_{t-1}$相乘，以忘掉其中的一些信息，二者相结合，便得到更新后的状态$C_t$。具体公式如下:
    $$
        i_t = sigmoid(W_i*[h_{t-1},x_t]+b_i) 
    $$
    $$
        \tilde{C_t} = tanh(W_c*[h_{t-1},x_t] + b_c)
    $$
    $$
        C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C_t} 
    $$
3. 输出门
    最后，LSTM需要计算最后的输出信息，该输出信息主要由元胞状态$C_t$决定，但是需要经过输出门进行过滤处理。首先要将元胞状态$C_t$的值规范化到[-1,1]，这通过tanh层来实现。然后依然由一个sigmoid层得到输出门参数$o_t$，最后将$o_t$和规范化后的元胞状态进行点乘，得到最终过滤后的结果。具体公式如下:
    $$
        o_t = sigmoid(W_o*[h_{t-1},x_t] + b_o)
    $$
    $$
        h_t = o_t \odot tanh(C_t)
    $$

## GRU

1. 更新门$z_t$
   $$
    z_t = sigmoid(W_z*[h_{t-1},x_t])
   $$
   
2. 重置门
   $$
   r_t = sigmoid(W_r * [h_{t-1},x_t])
   $$
3. 记忆体
   $$
   h_t = z_t{\odot}h_{t-1} + (1-z_t) \odot \tilde{h}_t
   $$
4. 候选隐藏层
   $$
        \tilde{h}_t = tanh(W * [r_t\odot{h_{t-1}},x_t])
   $$
   