# 基于V13B的Rank1优先方法筛选

本轮用户询问现成方法及自己的想法，仅做本地/文献研究，不连接服务器、不提交或修改训练。结果基于2026年10月7日09:24快照，文档更新时间不代表新远端进度。

## 现有首位检索缺口

V13B固定100轮mAP12.1738%、Rank1 28.5090%、Rank5 44.4965%、Rank10 52.1155%。6405有效query：

|首个正确身份排名|query数|
|---|---:|
|第1|1826|
|第2—5|1024|
|第6—10|488|
|第10之后|3067|

Rank1≥30%至少1922个首位正确，需要净多96个。1024个第2—5名query提供“已有线索但首位被占”的潜在空间，但不能保证全部可通过训练换序、不假定修正时不会损失原本正确query。

完整逐query正负相似度/错误首位文件尚未保存于V13B快照，不能由汇总值推断具体身份、光谱/camera错误分布或真实margin。后续固定模型错误审计需原checkpoint推理/已存特征，不能补造。

## 为什么优先改竞争样本来源

源码核对：V13B原P16K4每批随机16身份/64图，triplet在该批3个raw分支各自挖掘。每anchor通常只有15个其他身份当批参与；已有identity_full原型涉及全500身份，但在768维对平均中心竞争，不能等同于对最相似的错误图片实例竞争。正式检索是raw768+raw512拼接L2的1280余弦、93609全图库。

这些源码区别支持“扩大有信息实例竞争”的候选，尚未证实它是唯一Rank1瓶颈。旧三分支triplet已经有hard mining，不能把再说一遍“困难负样本”当作新方法；要改变哪些实例有机会进入学习。

## 现成方法优先级

### 1. Graph Sampling，CVPR2022：首选采样对照

在每epoch建立身份近邻图，batch由anchor身份及近邻身份组成，提前把容易混淆身份的真实图放进训练。原论文表3同QAConv/hard-triplet的采样比较，MSMT17→Market的PK Rank1 75.7、GS79.1（+3.4百分点）。不同数据/骨干，不外推WHU-MARS提升量；完整系统的其他大幅成绩不能归给采样本身。

适配V13B可首先保持原网络、全部loss、I2T、LR/100轮日程及1280推理，只改变身份选择；K4从原身份图池取图。渐进混合随机和近邻身份，保持500身份覆盖。原论文epoch batch数与本项目不同，需要明确匹配/记录批次、图像暴露、成功更新和AMP跳过；构图如增加前向需披露成本。若改三谱/异camera身份内图选，则是第二个变量，首轮不同时叠加。

来源：[Graph Sampling](https://openaccess.thecvf.com/content/CVPR2022/html/Liao_Graph_Sampling_Based_Deep_Metric_Learning_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.html)，[作者代码QAConv](https://github.com/ShengcaiLiao/QAConv)。

### 2. Cross-Batch Memory，CVPR2020：首选实例记忆候选

保存过去批次的实例embedding，扩展有信息正负样本来源。与原prototype不同，它保留某人某camera/光谱图的具体外观，而非只存一份身份均值。可用最终归一化1280实例特征并记录训练身份、camera/光谱、成功step/年龄；排除同camera候选、同身份不当负例，保留高相似错误实例。

B后期LR下降使“历史特征漂移较慢”的前提可能更合适，但必须测量，不能由小LR就认定旧实例与当前模型一致。早期预热、刷新/年龄上限、身份重复/场景占比、困难负例噪声与真实梯度均需诊断；只在成功optimizer更新后写记忆。

若在最终1280增加pair loss，监督空间和权重也改变。需要无记忆的同形式loss对照区分记忆与新度量几何，不能称纯XBM作用。原raw3分支损失宜先保留，避免重复V14整体替换风险。

来源：[XBM](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)，[作者代码](https://github.com/MalongTech/research-xbm)。

### 3. Circle Loss / Multi-Similarity：小范围监督对照

Circle按正负相似度偏离目标的程度赋权；MS对有信息pairs做挖掘和相似度加权。适合比较最终归一化空间的判别目标，不保证Rank1提升。V14度量整体替换、V15/V16跨camera排名均未自动带来收益，所以不以loss名字作为下一首选。

来源：[Circle Loss](https://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.html)，[MS Loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.html)。

### 4. 成对局部matcher / BIT：单独系统路线

利用query-gallery成对局部交互判断“形似但不同人”，更直接改变首位评分；前k候选精排是可能改编，但改变单图独立描述子余弦评分/无重排序主协议，不能混入现有正式结果。全图库约6亿query-gallery对，不可忽略成对成本。

来源：[BIT](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_BIT_Matching-based_Bi-directional_Interaction_Transformation_Network_for_Visible-Infrared_Person_Re-Identification_CVPR_2026_paper.html)；本地已有全文/先前文献核对支持其机制。本轮搜索未返回该特定标题的新匹配，不声称新完整PDF复验。

## 自己的方法方向：场景条件的实例近邻竞争

普通GS/XBM为现成训练基线，不直接作为CVPR创新。针对本数据可研究：

1. **竞争关系按实际光谱/空地条件形成。** 用训练集真实观测的实例关系建立分方向邻居，避免某身份只有一个均值抹掉其易混淆的特定视角。不存在的空中观测不填补；500身份均保留。
2. **瞄准可靠正确实例与最相似错误实例的差。** Rank1需要最佳合法正例超过最高合法负例（等分另按稳定排序）；单纯拉近全部同身份/所有困难部位可能增加噪声。用多正例软权重保证异camera覆盖，保留原正例学习，不把“只挑最容易正例”当成答案。
3. **以真实竞争者刷新训练机会。** 从实例近邻挖掘，再将其真实图片逐步带入batch；陈旧实例关系用于选样，当前前向提供梯度。每批保留随机身份，监测困难程度、采样覆盖、正负margin和特征方向变化。
4. **避免V16全拒绝。** 优先软权重/连续难度，没有可靠候选时保留基础监督；同时记录原始候选、筛选各步、有效gradient，不能只用最苛刻多重硬门槛。

训练camera/光谱可用于形成合法关系，正式视觉推理仍不读身份、camera、光谱标签或图库数据。官方测试错误只作预登记诊断，不用于选择训练邻居、调权重或反复选方案。

这只是可研究的区别，需与普通GS、普通XBM、全局身份中心近邻/场景条件实例近邻等匹配对照证明收益与贡献，不能预先宣称新颖。

## 建议实验顺序及授权边界

首先选“V13B配方＋渐进近邻身份采样”单因素对照：基础网络/原loss/初始化/尾部余弦/推理保持；独立批准Stage1和零辅助记忆开始，固定末轮，记录真实预算。它的变化比V15/V16整体重构更容易分析。

之后单独比较“V13B＋实例记忆竞争”，搭配同形式无记忆对照；Circle/MS及新场景条件方法在机制诊断成立后考虑，不把多个改动同时叠加。Rank1重点也须同时看mAP与六场景，避免跨过30%但破坏已取得的mAP12%。

本轮没有新训练预算/部署授权，不从B100或B50续训、不改旧任务、不连接服务器。V13B已有主模型/完整状态/证据本地核验待补。用户CVPR原则保持：现成模块为基线，自己的方法靠明确问题、机制差异、有效消融/多seed/跨数据证据论证。
