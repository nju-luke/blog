---
title: tensorflow中多模型复用（多重加载）的问题
tags: [技术杂烩,tensorflow]
categories: 技术修炼
---

在tensorflow中遇到下面两个问题：

1. tflearn中，一个模型的输出作为另一个模型的输入，此时第二个模型的输入将第一个模型的输入也计算在内

2. 同一个模型利用多个不同初始化参数得到模型，在同一个文件内加载这些模型时，第二次加载时默认带上了第一次加载时参数的错误。

   ```
   tensorflow.python.framework.errors_impl.NotFoundError: Key model_0/hed/conv1/conv1_1/biases not found in checkpoint
   ```

以上第二个错误在[Loading Multiple models into the same session of tensorflow](https://github.com/tensorflow/tensorflow/issues/3270)得到解决办法。原理也相对较简单，即通过限定restore时参数的范围。

多模型复用、多重加载时采用以下方法：

1. 通过`tf.variable_scope()`限定每个模型的参数名称。若不做限定，第二次加载时将出现参数已存在的错误

2. restore建立saver之前，通过名称限定参数

   ```python
   model_vars = [k for k in tf.global_variables() if k.name.startswith(name)]
   saver = tf.train.Saver(model_vars)
   saver.restore(sess,model_path)
   ```

   ​