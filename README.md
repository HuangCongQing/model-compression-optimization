<!--
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-10-16 10:28:52
 * @LastEditTime: 2022-12-20 18:26:31
 * @FilePath: \model-compression-optimization\README.md
-->
# model-compression-optimization
model compression and optimization for deployment for Pytorch, including knowledge distillation, quantization and pruning.(知识蒸馏，量化，剪枝)



## 1 Pruning(剪枝)


#### 算法总表
| **Pruning Method** | **Code location** | **Docs** | **Remark** |
| --- | --- | --- | --- |
| **01开山之作：Learning Efficient Convolutional Networks Through Network Slimming (ICCV2017)** | code: [pruning/01NetworkSlimming](pruning/01NetworkSlimming) <br> code reference: <br>[link1]( https://github.com/foolwood/pytorch-slimming) <br>  [link2](https://github.com/Eric-mingjie/network-slimming)| [docs](https://www.yuque.com/huangzhongqing/pytorch/iar4s1) | placeholder |
| **02【ICCV2017】ThiNet** | code: [1pruning/02ThiNet](1pruning/02ThiNet) <br> code reference: <br> https://github.com/SSriven/ThiNet | [docs](https://www.yuque.com/huangzhongqing/lightweight/pnzhr3tb8wfdciep#Kownj) | 1 |
| **03【CVPR2020】HRank** | code: [1pruning/03HRank](1pruning/03HRank) <br> code reference: <br>[link](https://github.com/lmbxmu/HRank) | [docs](https://www.yuque.com/huangzhongqing/lightweight/xqks1lrte52moirq#dRSJK) | placeholder |
| **Coming...** | 1 | 1 | 1 |


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


#### 算法总表

| **量化 Method** | **Code location** | **Docs** | **Remark** |
| --- | --- | --- | --- |
| **Coming...** | 1 | 1 | 1 |



## 3 knowledge distillation(知识蒸馏)

#### 算法总表
| **KD Method** | **Code location** | **Docs** | **Remark** |
| --- | --- | --- | --- |
| **01开山之作： Distilling the knowledge in a neural network（NIPS2014）ndom** | code: [3distillation/01Distilling the knowledge in a neural network](3distillation/01Distilling_the_knowledge_in_a_neural_network)<br>code reference: https://github.com/Eli-yu-first/Artificial_Intelligence | https://www.yuque.com/huangzhongqing/lightweight/lno6i7 | 1 |
| **02  Channel-wise Knowledge Distillation for Dense Prediction（ICCV2021）** | code: [3distillation/02SemSeg-distill](3distillation/02SemSeg-distill) <br> code reference: https://github.com/irfanICMLL/TorchDistiller/tree/main/SemSeg-distill | https://www.yuque.com/huangzhongqing/lightweight/dourdf2ogh9y1cx9#VHZBv | 1 |
| **Coming...** | 1 | 1 | 1 |



#### 01开山之作： Distilling the knowledge in a neural network（NIPS2014）


docs: https://www.yuque.com/huangzhongqing/lightweight/lno6i7

code: [3distillation/01Distilling the knowledge in a neural network](3distillation/01Distilling_the_knowledge_in_a_neural_network)


code reference: https://github.com/Eli-yu-first/Artificial_Intelligence


#### 02  Channel-wise Knowledge Distillation for Dense Prediction（ICCV2021）


docs: https://www.yuque.com/huangzhongqing/lightweight/dourdf2ogh9y1cx9#VHZBv

code: [3distillation/02SemSeg-distill](3distillation/02SemSeg-distill)


code reference: https://github.com/irfanICMLL/TorchDistiller/tree/main/SemSeg-distill


## 4 NAS神经网络搜索(Neural Architecture Search,简称NAS)

video:
* 神经网络结构搜索 Neural Architecture Search 系列:https://space.bilibili.com/1369507485/channel/collectiondetail?sid=788500
* PPT: [4NAS/NAS基础.pptx](4NAS/NAS基础.pptx)



#### 算法总表
| **NAS Method** | **Code location** | **Docs** | **Remark** |
| --- | --- | --- | --- |
| **01 DARTS(ICLR'2019)【Differentiable Neural Architecture Search 可微分结构】—年轻人的第一个NAS模型** | code: [4NAS/01DARTS(ICLR2019)/pt.darts](4NAS/01DARTS(ICLR2019)/pt.darts) <br> code reference: <br> https://github.com/khanrc/pt.darts | hthttps://www.yuque.com/huangzhongqing/lightweight/esyutcdebpmowgi3 | video:【论文解读】Darts可微分神经网络架构搜索算法:https://www.bilibili.com/video/BV1Mm4y1R7Cw/?vd_source=617461d43c4542e4c5a3ed54434a0e55 |
| **Coming...** | 1 | 1 | 1 |


#### 01 DARTS(ICLR'2019)【Differentiable Neural Architecture Search 可微分结构】—年轻人的第一个NAS模型

doc：https://www.yuque.com/huangzhongqing/lightweight/esyutcdebpmowgi3

code: [4NAS/01DARTS(ICLR2019)/pt.darts](4NAS/01DARTS(ICLR2019)/pt.darts)
code reference:：https://github.com/khanrc/pt.darts
video:【论文解读】Darts可微分神经网络架构搜索算法:https://www.bilibili.com/video/BV1Mm4y1R7Cw/?vd_source=617461d43c4542e4c5a3ed54434a0e55

#### 02 TODO

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
