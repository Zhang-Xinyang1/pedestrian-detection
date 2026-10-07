# CVPR · V13B 行人重识别完整复现包

> 上传进度：代码、已校验的第100轮正式模型和预训练权重已准备就绪。第50轮模型与完整训练状态正在传输，稍后将补充到同一仓库。本地完整数据已完成逐图校验。

本目录整理自 `SCNET_REPAIR_V13_LR100_AB_20261005` 的 V13B 版本。研究任务是 WHU-MARS 多光谱空地行人重识别（AS-ReID）。V13B 使用 CLIP ViT-B/16、模态提示、全身份原型、L2-SP、可拒绝局部匹配监督，并在第 41–100 轮采用余弦学习率尾部。

原训练与模型实现保持发布包原始内容，新增的整理、训练和评价入口放在 `scripts/`。仓库地址：[Zhang-Xinyang1/pedestrian-detection](https://github.com/Zhang-Xinyang1/pedestrian-detection)。

## 已保存的结果

| 模型 | mAP | Rank1 | 用途 |
|---|---:|---:|---|
| V13B，第 100 轮 | 12.1738% | 28.5090% | 预登记固定 100 轮正式结果；最高已评价 mAP |
| V13B，第 50 轮 | 11.9962% | 29.1335% | 已评价检查点中的最高 Rank1，辅助分析 |

两项成绩均来自同一 seed1 正式训练的原始评价 JSON。评价每 10 轮执行一次；第 50 轮不能替代固定第 100 轮主结果。mAP 达到 12%，Rank1 尚未达到 30%。这是研究复现包，文件夹名称不代表 CVPR 录用。

训练集 **92,133 张、500 个身份**；query **6,405 张**；gallery **93,609 张**。RGB / IR / Thermal 为数据目录名称。评价使用 raw768 + raw512 拼接、整体 L2 归一化、1280 维特征、排除所有同 camera 图像，无重排序。空中到空中无合法正例的方向按原协议记录为 null。

## 文件布局

```text
CVPR/
├── code/v13b/                    原始发布包、模型、loss、采样器、训练器、测试、离线 wheels
│   └── package/upstream/         CLIP-ReID 原始代码与 MIT 许可证
├── pretrained/                   原始 CLIP 权重与批准的 Stage1 第120轮模型
├── artifacts/V13B/
│   ├── production100/            第100/50轮模型、完整状态、全局/局部记忆、评价曲线与审计
│   └── *.json / *.log            原始环境、训练过程与本地下载校验
├── datasets/
│   ├── WHU-MARS/                 本地完整图像：train / query / test / querymm
│   ├── WHU-MARS-v13b.zip          当前实验数据的完整本地归档
│   ├── DATASET_MANIFEST.json      数量、身份数和归档校验
│   └── IMAGE_SHA256SUMS.txt        每张图像的校验值
├── scripts/                      数据校验、V13B训练入口、固定模型评价入口
├── docs/                         原版本说明、数据协议与第三方许可证
├── environment.yml               与正式服务器配套的 Python / PyTorch 环境
└── .gitattributes                模型与离线 wheels 的 Git LFS 配置
```

`querymm` 为本地额外数据，未参与 V13B 正式评价。原始 2337-ID 大数据 ZIP 与本次实验的 500-ID 训练划分不同，因此这里重新归档了当前实际使用的图像，避免拿错数据版本。

## 环境

原正式训练：Linux、Python 3.12.7、PyTorch 2.6.0 + CUDA 12.4、torchvision 0.21.0、单张 A800 80GB。CPU/Windows 可执行文件校验；原模型构造与训练/推理需要 CUDA。Windows RTX3050 工程检查与正式全量结果分开记录。

```bash
conda env create -f environment.yml
conda activate cvpr-v13b
```

如果已有与上述版本匹配的 GPU 环境，可安装其余锁定依赖：

```bash
python -m pip install -r code/v13b/requirements-lock.txt
```

离线 wheels 位于 `code/v13b/wheels/`，包含原 Python 3.12 Linux 依赖，PyTorch/CUDA 由原服务器镜像提供。跨平台运行请使用当前平台匹配的依赖。

## 数据准备与核验

本地整理目录已包含全部图像，可直接检查：

```bash
python scripts/prepare_data.py --verify-images
python scripts/verify_release.py
```

从 Git 克隆后，数据图像与归档默认不会下载。原 WHU-MARS README 明确要求不再分发数据，完整协议见 `docs/WHU-MARS_Agreement.pdf`、`docs/source/WHU-MARS_original_README.md`。获得数据使用权限后，将本地归档放入 `datasets/WHU-MARS-v13b.zip`，在空的数据目录中执行：

```bash
python scripts/prepare_data.py --extract --verify-images
```

程序检查归档 SHA256、路径安全和原实验划分；已有图像会拒绝覆盖。这里的 `.gitignore` 仅排除原始数据与新实验输出，模型及已有正式实验记录可以提交。

## 训练

先进行原始输入/Stage1/数据协议核验，不启动训练：

```bash
python scripts/train_v13b.py --mode check
```

进行原始 4 轮小规模工程 smoke：

```bash
python scripts/train_v13b.py --mode smoke
```

按原正式流水线复现：环境、完整数据契约、CPU/CUDA检查、独立smoke、独立100轮训练、正式评价和结果核验。

```bash
python scripts/train_v13b.py --mode pipeline
```

原流水线严格检查 Python3.12 / torch2.6 / torchvision0.21 / A800。已准备合适依赖目录时可加 `--site /path/to/site`。其他 CUDA 硬件可使用保持相同方法与参数的训练入口：

```bash
python scripts/train_v13b.py --mode train --output outputs/v13b100_seed1
```

上述入口均采用原批准 Stage1 和零记忆独立启动，不使用旧版本模型续训。`train` 模式只运行原训练主程序，完整预检查与训练后工程核验由 `pipeline` 模式执行。数据、seed1、P16K4 batch64、100轮预算、所有loss和学习率参数保持 V13B。

`training_state_latest.pth` 保存模型、optimizer、scheduler、AMP、RNG、全局/局部记忆和 epoch100 账目；原代码有状态恢复函数，原正式流水线没有自动续训 CLI。不能把只含参数的主模型当作完整训练状态。

## 固定模型评价

```bash
python scripts/evaluate_v13b.py --epoch 100
python scripts/evaluate_v13b.py --epoch 50
```

默认 batch32，使用与原评价相同的 resize/normalize 和检索指标。结果写入 `outputs/`，既有正式评价文件保持原内容。以下仅验证模型加载和前向，不产生正式检索成绩：

```bash
python scripts/evaluate_v13b.py --epoch 100 --max-images 2 --batch-size 1 --workers 0
```

全量评价需处理 100,014 张图像并进行全图库排序，运行时间取决于设备。原始正式分数见 `artifacts/V13B/production100/eval_stage2_100.json`；逐轮曲线见 `docs/V13B_RESULTS.csv`。

## Git 上传

权重及离线 wheels 已配置 [Git LFS](https://git-lfs.com/)。Git 保存指针，完整二进制文件由 LFS 上传和下载。原始数据按数据协议保留本地；SSH凭据、密码、虚拟环境和临时文件不进入仓库。

```bash
git lfs install --local
git add .
git commit -m "Organize complete V13B reproduction package"
git branch -M main
git remote add origin https://github.com/Zhang-Xinyang1/pedestrian-detection.git
git push -u origin main
```

已经配置 `origin` 时跳过 `git remote add`。克隆时安装 Git LFS 并运行 `git lfs pull`，即可取回权重。托管方的 LFS 容量与传输限额以其账号配置为准。

## 来源

原发布包校验保存在 `code/v13b/PAYLOAD_SHA256.json` 与 `code/v13b/package/SHA256SUMS`。第100/50轮模型及完整状态与原服务器校验值逐一比对；下载记录见 `artifacts/V13B/LOCAL_DOWNLOAD_VERIFICATION.json`，整理后的文件清单见 `RELEASE_MANIFEST.json`。

原服务器每10轮保存模型；本包保留正式第100轮、第50轮Rank1峰值和完整末轮训练状态。其余中间模型保留在原服务器，原 `ARTIFACT_SHA256.json` 清单及所有逐轮评价/审计原样保存。`scripts/verify_release.py` 检查本包实际包含的文件，并明确列出未复制的中间模型。

第三方 CLIP-ReID 代码使用其原 MIT 许可证；原数据使用协议单独保留。本整理没有为第三方数据或代码重新授予许可。
