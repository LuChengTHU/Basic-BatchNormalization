# Basic-BatchNormalization
本次实验中，我用 TensorFlow实现了 BN（Batch Normalization,）层与 CNN、MLP的结合，并比较了 CNN与 MLP的区别，初步探讨了 BN层对训练的影响。

关于 BN层的实现，调用了 TensorFlow的部分函数，并自己实现了均值、方差的动态更新和 one by one策略
