# CSF原型对照v3

五组：control、identity、balanced_modality、six_scene、csf。
同一已验证Stage1 checkpoint，seed1，P16K4，120 epoch，每10 epoch评估，无指标早停。
四个原型组固定权重0.1、温度0.07、重构权重0.1；control无原型loss。
所有组均维护同一结构的非参数memory作诊断，原型组使用一致的样本/候选有效性规则：
当前场景有观察、三个模态有观察、加性估计满秩。模型检索保持1280维。
这是一项匹配条件下的原型几何对照；不能将受共同候选mask约束的identity方法称为原版UAD。
后续可另做完整候选identity/UPCL-style参照，但需分别报告覆盖差异。
csf_smoke.sbatch在一张A800上依次验证五组并核验相同起点。
服务器预检与全部GPU smoke通过后，才提交csf_stage2.sbatch。
新增模块CPU/CUDA测试已通过，尚无生产效果结论。
