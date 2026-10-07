# V13A／V13B：后期学习率单因素对照，独立固定100轮

用户最新指定两区各提交一版、每版100epoch，不等已有四区任务结束。该授权覆盖当前两区各一张A800资源内启动独立实验，旧V12／V10-80任务和正式证据保持。正常训练确认后关闭SSH，不轮询、不创建监控。

## 两版唯一实验变量

共同基础为已核验V10：CLIP ViT-B/16、批准Stage1、零全局／局部记忆、seed1、92133训练图／500身份、原增强、P16K4 batch64、Adam、I2T1.0、identity_full原型、L2-SP0.001、局部可拒绝匹配和KL保持。新参数0、新视觉前向0。

|版本|区域|1–40轮|41–100轮|
|---|---|---|---|
|V13A，对照|华东四区|V10原日程，21–40轮base LR5e-7|固定base LR5e-7|
|V13B，候选|华东五区|与A相同|从第40轮5e-7余弦下降到第100轮1e-8|

bias倍率2保持。余弦公式为`1e-8 + (5e-7 - 1e-8) * (1 + cos(pi*(epoch-40)/60))/2`，从第41轮生效。这组只检验后期继续固定LR是否与回落有关；不能归因到batch、I2T或新监督，也不保证降LR足够达到30% Rank1。

使用同一份代码ZIP，以`--arm A`／`--arm B`选择。没有分别维护容易漂移的两份实现。对照A在生产100轮的scheduler与原V10日程逐步一致；前40轮A/B相同。真实跨容器结果可能有数值差异，需比较相同阶段模型指纹、数据／采样／初始化和AMP账目，不能提前保证两区字节完全复现。

## 启动

Python3.12.7／PyTorch2.6.0+cu124／torchvision0.21.0，单张A80080GB。共同ZIP：`SCNET_REPAIR_V13_LR100_AB_20261005.zip`。

四区上传ZIP和A脚本后：

```sh
PROJECT_ROOT=/work/home/luhanning/prvc PYTHON_BIN=/opt/conda/bin/python sh /work/home/luhanning/prvc/SCNET_START_V13_LR100_A.sh
```

五区上传同一个ZIP和B脚本后：

```sh
PROJECT_ROOT=/root/private_data/prvc PYTHON_BIN=/opt/conda/bin/python sh /root/private_data/prvc/SCNET_START_V13_LR100_B.sh
```

新run分别为：

```text
scnet_v13_lr100_ab_runs/V13A_v10_constant_tail_fixed100_seed1
scnet_v13_lr100_ab_runs/V13B_v10_cosine_tail_fixed100_seed1
```

正式日志`production100.log`、产物`production100`、主模型`ViT-B-16_100.pth`。已有同名run或对应`ACTIVE_A.lock`／`ACTIVE_B.lock`则拒绝重复启动，不删锁。禁止把旧V10第40／80轮或V12模型当作该独立实验起点。

程序核对包SHA、环境、Stage1、全训练元信息、原P16K4预检查及固定100轮契约，然后执行CPU／CUDA优化日程、完整状态恢复、局部匹配及原损失行为检查。4轮真实小规模训练独立从Stage1开始，压缩学习率尾部边界到第3轮并在第4轮到达终值，验证实际optimizer应用A／B区别；CPU／CUDA日程检查另外覆盖全部1–100轮、40／41边界与bias×2。smoke不是生产日程或检索结果，正式阶段再次从Stage1和零记忆开始。

## 记录与核验

每轮保存`learning_rate_epoch_history.json`，检查实际optimizer各组LR。每10轮保存主模型及最新完整状态`training_state_latest.pth`，含模型、optimizer、scheduler、AMP、CPU／CUDA／NumPy／Python RNG、全局／局部记忆及epoch账目，原子替换。状态恢复函数已通过下一次随机optimizer更新一致性检查；正式pipeline入口仍独立Stage1启动，不自动断点重试或启动新任务。不要把只下载的主模型称为无损续训状态。

固定第100轮比较A/B总体、六场景及方向query分母；第40轮作为改动前的检查点，第60／80轮用于趋势诊断。正式检索保持6405query／93609gallery、raw768＋raw512拼接后整体L2的1280维、排除全部同camera、无重排序。空中到空中无合法跨相机正样本格为null。

当前主目标是第100轮mAP≥12%、Rank1≥30%，两项需同一checkpoint同时达到。论文UAD11.0%／29.5%、历史10.83%／28.40%分开报告。`FIXED100_TARGET_REPORT.json`与工程VERIFICATION独立，工程PASS不会把未达目标改成成功。不能挑第50轮等测试峰值替代末轮；单seed不证明稳定性。

`collect_results.sh --arm A`／`--arm B`只打包原产物，不覆盖VERIFICATION／ARTIFACT_SHA256。末轮模型分别在本地保存`v13a-ViT-B-16_100.pth`、`v13b-ViT-B-16_100.pth`；完整状态可另行下载，收集包默认排除全部`.pth`。

## 当前证据与研究边界

V10独立80轮服务器末轮为11.4446%／26.9945%，低于自身第40轮11.8654%／28.7432%；旧任务结果与新A对照不能当作120轮结果。V12当前40轮结果另行收集。这组没有新增配对监督，避免将不同方法效果与尾部LR混在一起。

当前V10原日程是第21轮以后固定5e-7；UAD为不同主干／SGD／PKM／cosine配置。当前两版只改变尾部LR，不声称复制UAD或已经定位全部早期饱和原因。前40轮采样和loss维持原协议，后期若只保住原成绩但未达到12%／30%，应如实报告。
