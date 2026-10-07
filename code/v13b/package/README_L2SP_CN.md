# CSF完整身份监督＋起点锚定 / 固定40轮 v1

本包是独立新方案，不覆盖v1/v2/v3历史包或181267产物。

## 本轮实际改变

- 方法：`csf_full_identity`。所有已观测identity memory均可参与原型判别；满秩identity使用
  identity+modality residual+view residual，未满秩identity回退到已观测global identity center。
  不伪造未观测view，也不改变输入采样。
- 优化：前20 epoch沿用181267；epoch21固定降到5e-7，固定训练40 epoch。
- eval/checkpoint：10/20/30/40，末轮40固定为主checkpoint。
- 保留CLIP、I2T=1.0、CSF权重0.1、temperature0.07、reconstruction0.1、L2-SP系数0.001。
- 不包含QA Adapter、teacher、额外dropout、lambda扫描或自动延长60 epoch。

这是一个联合验证方案：它同时检验“完整身份监督”与“尾段低LR精修”。epoch30可与181267作调度诊断；
epoch40是新预算，不能伪造历史epoch40 UAD门槛。第一30轮仍按10/20/30门槛记录，epoch40另与完整UAD 10.83/28.40及正式12/29比较。

## v3测试隔离修复

v2仅部署，未提交：服务器无GPU登录节点的CPU回归测试发现，真实训练函数末尾
torch.empty(device='cuda')未被测试替身拦截。v3在测试中将这个空诊断tensor映射到CPU；
不修改训练函数。v2包和所有预检失败记录保留。
v3的实际训练代码与v2逐字一致，仍只修复训练计数与诊断均值；新增CPU替身仅属于测试。

## v2修复与历史边界

181057（v1）在2026-10-03 12:13:09北京时间以FAILED/1:0结束。
CUDA正则、诊断不干扰梯度和原型测试通过；4epoch smoke训练完成，随后审计报错
`L2SP batch budget differs`，没有smoke验证标记，也没有进入正式30轮。
根因是eval循环复用了n_iter：smoke第4轮实际6个train batch，被记录成12个eval batch，
平均正则项的分母也被覆盖。v2使用独立训练计数，修复审计账目和该诊断均值，
不改训练loss、反向、optimizer、学习率或系数。verify_results严格检查原样保留。
新增test_l2sp_epoch_accounting.py运行实际训练循环，故意使用6个train/12个eval batch；
对v1重现失败，对v2验证训练计数和正则均值正确。GPU allocation内再次运行该测试。
v1包、181057日志和全部smoke产物保留；v2从批准Stage1新起点和零memory开始。

本文件为新版本执行说明；继承的v3/原型/Adapter以外包文档仅为历史参考。
以原型CSF为基线，单因素增加L2-SP思想的起点锚定。不包含QA Adapter、额外gate、
dropout、新的prompt或双次teacher forward。L2-SP是已有正则基线，不作为论文新颖性。

## 冻结配方

L_total = L_original_CSF + 0.001 * 0.5 * sum_visual_params ||theta - theta_Stage1||^2。

系数预先固定0.001。采用全元素sum，无batch/层/元素数归一化，不运行系数扫描。
完整锚定 model.named_parameters 中 image_encoder.*，包含CLIP视觉projection；
不锚定分类器、bottleneck、text/prompt或memory。每个参考tensor detach/FP32 clone，
放在与对应参数相同device，不注册到model，不进入optimizer，新增可训练参数为0。
起点来自批准Stage1 strict加载后的同构模型，初始penalty必须严格为0。
参考hash需与批准的初始视觉参数一致，训练后参考hash不能改变。
既有Adam weight_decay照旧保留，学习率warm-up10和milestones[60,100]保持原样。
I2T=1.0，CSF weight0.1，temperature0.07，reconstruction0.1，seed1。

不修改原图/标签/split/增强/P16K4 batch64；WHU同camera全部排除，视觉检索始终1280维。
末轮30固定为screening主结果，10/20/30checkpoint均保存供诊断，不按测试峰值选取。
10/20/30中至少两个检查点同时严格超过UAD对应mAP/Rank1，才gate PASS。
同时报告相对原型CSF的末轮及中期退化，gate PASS本身不证明正则有效。

## 诊断

每epoch每300 batch抽取一次（含首batch），不额外读取图像、不改变采样和RNG。
沿用原loss_fn计算真实loss；重构相同ID/triplet/I2T loss仅用于观测，并检查两者数值一致。
在最后ViT block的mlp.c_proj.weight上测各weighted loss梯度范数和task-reg梯度余弦，
autograd.grad不修改Parameter.grad。梯度观测有额外少量反向成本，不作为训练信号。
记录raw768/raw512范数、拼接前512平方范数比例以及每epoch完整参数漂移和penalty。
必须通过观测前后实际梯度一致性CUDA测试。

正式全6405query×93609gallery的1280指标仍用未改动whu_adapter/whu_metrics计算。
另在固定每scene最多32个均匀分布query（共最多192个）和完整gallery上，计算768、512、1280
三种表示的诊断指标，采用同camera排除，并显式diagnostic_only。
单分支结果不替代正式指标，不用它们选择测试checkpoint或修改最终1280接口。

## 审计与单卡

保存l2sp_initialization.json、l2sp_audit.json、training_diagnostics.json、
branch_diagnostics_stage2_{010,020,030}.json和每期checkpoint/hash/参数指纹。
verify_results保留原来的严格加载、参数/frozen/optimizer范围、memory步数、完整预算及SHA。
额外检查锚定范围/系数/引用hash、参考不变、末轮实际漂移与从源checkpoint重算一致、
正则梯度非零、各期检查点、诊断scope和gallery不变。

唯一新入口 l2sp_screen30.sbatch，单张A800、d1n41a16g01、vip_gpu_a800_scwc189、gpugpu。
同一allocation依次CUDA tests→4epoch smoke→verify→新起点30epoch→gate→verify。
smoke失败不进入30轮；smoke和production独立output，均从批准Stage1重新开始和reset memory。
不要重复提交；提交后只报告JobID/初始状态/用户检查时间，不持续轮询。

参考：[L2-SP / ICML2018](https://proceedings.mlr.press/v80/li18a.html)。
