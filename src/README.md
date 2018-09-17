# facenet 的 tripletloss 版本 和 softmax 版本的使用说明

## 1. tripletloss 

**说明：tripletloss 版本的代码，已经改成了多 GPU 的版本，训练代码已经验证代码都已经修改完成。**

 - 1.单 gpu 版本使用的代码文件为：
    - 启动训练的 shell 脚本文件：train.sh
    - 使用的训练过程的 python 文件：train_tripletloss.py
    - 验证训练完成模型的 python 文件：validate_on_lfw.py

 - 2.多 gpu 版本使用的代码文件为：
    - 启动训练的 shell 脚本文件：train.sh
    - 使用的训练过程的 python 文件：train_multi_gpu_2.py（训练过程，使用的是 每个 gpu 跑一个 batch，而不是将每个 batch 拆分来做并行跑）
    - 验证训练完成模型的 python 文件：validate_on_lfw_mul.py

 - 3.（实验版本）多 gpu 版本使用的代码文件：
    - 启动训练的 shell 脚本文件为： train_new.sh
    - 使用的训练 python 文件：train_multi_gpu_new.py（训练过程中，将每个batch拆分成若干份，在不同的 gpu 中进行并行计算）
    - 验证文件：validate_on_lfw_mul.py

**比较的方面主要为以下几个方面：**

 - 运行 20 个 epoch 的时间
 - 生成的模型的准确度
 - 训练是否调用了全部的 gpu（这个已解决，全部的 gpu 都已经使用上了）   

**训练好的模型存放在 240 服务器上的位置：**

 - 单 gpu： 20180906-111143
 - 双 gpu： 20180910-140144
 - 三 gpu： 20180910-155145

**对应的训练时间 和 准确度如下：**

 - 单 gpu： 20 个 epoch ，106.327 分钟，准确度：0.71150
 - 双 gpu： 10 个 epoch，94.379 分钟，准确度：0.60217
 - 三 gpu： 11 个 epoch，106 分钟，准确度：0.622667

**初步得到结论：**

随着 gpu 的个数的增多，所用的时间越来越长，具体原因有待考察。

参考的 cosface 对 tripletloss 进行的更改。最终版本为 train_multi_gpu_2.py 。其中的 network.inference 是在 gpu 中进行分布式计算的，求 tower_loss 也是在 gpu 中进行的，最终的 total_loss 和 gradient 是在 cpu 中计算求均值的。

**随着 gpu 个数的增多，时间也相应增多的原因可能如下：**

 - 多卡带来了数据交换，导致性能下降。
 - 多卡之间，因为需要等待最慢的卡计算完成，所以等待时间会长一些。
 - 多卡对于小数据集的优势并不是很明显。
 - 查阅资料得知，增多了 gpu 之后，每个训练时候的 batch 需要根据 gpu 数量的增多而减小。（待考察）


## 2. softmaxloss 版本

**说明：softmax 版本已经更改成了多 gpu 运行的版本，因为训练过程会自动计算对应的准确度，所以无需再重新构建 evaluate 代码。**

 - 1.单 gpu 运行所需代码：
    - 训练出来的模型存储位置：20180914-130324
    - 训练的 shell 脚本：softmax_train_sg.sh
    - 训练的 python 代码： train_softmax.py

 - 2. 双 gpu 运行所需代码：
    - 训练出来的模型存储位置：20180913-192527
    - 训练的 shell 脚本：softmax_train.sh
    - 训练的 python 代码：softmax_mg.py

**对应的时间和准确度如下：**

 - 单 gpu ：20 个 epoch ，时间 144 min，准确度：0.91317
 - 双 gpu ： 90 个 epoch，时间 875 min，准确度：0.95167

## 3. 问题

在 softmax 的版本改版过程中，和 KP 商量了几次之后，意识到一个巨大的问题，是否我们在训练的时候，放到 每个 gpu 上进行训练的数据，是同一份 batch 。我们的目的是，将不同的 batch 放到不同的 gpu 上进行计算以达到加速的效果。所以说，这个要仔细考察一下。

20180917，现在有点眉目了，应该是如我们所料，最坏的结果，我们之前计算的时候，每次 gpu 循环，拿到的数据是几个 gpu 是一样的，也就是，同时拿到数据的 gpu 训练的数据是一样的，计算的多余了。T0T

## 4. 接下来的任务

接下来的任务是，我们对之前的 cosface 进行验证，对比验证，多个 gpu 并行训练和 单个 gpu 的并行训练。比较的指标如下：

 - 20 个 epoch 运行时间。
 - 每个 batch 运行时间。
 - 训练完成的模型准确度。