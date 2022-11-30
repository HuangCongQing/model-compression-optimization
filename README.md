<!--
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-10-16 10:28:52
 * @LastEditTime: 2022-11-26 23:29:16
 * @FilePath: \model-compression-optimization\README.md
-->
# model-compression-optimization
model compression and optimization for deployment for Pytorch, including knowledge distillation, quantization and pruning.(知识蒸馏，量化，剪枝)



## 1 pruning(剪枝)

#### 01 Learning Efficient Convolutional Networks Through Network Slimming (ICCV2017)
docs: https://www.yuque.com/huangzhongqing/pytorch/iar4s1

code: [pruning/01NetworkSlimming](pruning/01NetworkSlimming)

code reference:
>* https://github.com/foolwood/pytorch-slimming
● support for Vgg
>* https://github.com/Eric-mingjie/network-slimming
● We also add support for ResNet and DenseNet.



#### 02 TODO




## 2 quantization(量化)


#### 01 TODO

## 3 knowledge distillation(知识蒸馏)


#### 01开山之作： Distilling the knowledge in a neural network（NIPS2014）


docs: https://www.yuque.com/huangzhongqing/lightweight/lno6i7

code: [3distillation/01Distilling the knowledge in a neural network](3distillation/01Distilling_the_knowledge_in_a_neural_network)


code reference: https://github.com/Eli-yu-first/Artificial_Intelligence


## 4 NAS神经网络搜索(Neural Architecture Search,简称NAS)

video:
* 神经网络结构搜索 Neural Architecture Search 系列:https://space.bilibili.com/1369507485/channel/collectiondetail?sid=788500
* PPT: [4NAS/NAS基础.pptx](4NAS/NAS基础.pptx)

#### 01 DARTS(ICLR'2019)【Differentiable Neural Architecture Search 可微分结构】—年轻人的第一个NAS模型

doc：https://www.yuque.com/huangzhongqing/lightweight/esyutcdebpmowgi3

code: [4NAS/01DARTS(ICLR2019)/pt.darts](4NAS/01DARTS(ICLR2019)/pt.darts)
code reference:：https://github.com/khanrc/pt.darts
video:【论文解读】Darts可微分神经网络架构搜索算法:https://www.bilibili.com/video/BV1Mm4y1R7Cw/?vd_source=617461d43c4542e4c5a3ed54434a0e55

### 02 TODO

## TODOlist










## License

Copyright (c) [双愚](https://github.com/HuangCongQing). All rights reserved.

Licensed under the [MIT](./LICENSE) License.



---


微信公众号：**【双愚】**（huang_chongqing） 聊科研技术,谈人生思考,欢迎关注~

![image](https://user-images.githubusercontent.com/20675770/169835565-08fc9a49-573e-478a-84fc-d9b7c5fa27ff.png)

**往期推荐：**
1. [本文不提供职业建议，却能助你一生](https://mp.weixin.qq.com/s/rBR62qoAEeT56gGYTA0law)
2. [聊聊我们大学生面试](https://mp.weixin.qq.com/s?__biz=MzI4OTY1MjA3Mg==&mid=2247484016&idx=1&sn=08bc46266e00572e46f3e5d9ffb7c612&chksm=ec2aae77db5d276150cde1cb1dc6a53e03eba024adfbd1b22a048a7320c2b6872fb9dfef32aa&scene=178&cur_album_id=2253272068899471368#rd)
3. [清华大学刘知远：好的研究方法从哪来](https://mp.weixin.qq.com/s?__biz=MzI4OTY1MjA3Mg==&mid=2247486340&idx=1&sn=6c5f69bb37d91a343b1a1e7f6929ddae&chksm=ec2aa783db5d2e95ba4c472471267721cafafbe10c298a6d5fae9fed295f455a72f783872249&scene=178&cur_album_id=1855544495514140673#rd)
