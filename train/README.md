# PPO 训练脚本

本目录保存平衡重叉车视觉导航与末段对准策略的训练脚本。脚本均基于
Stable-Baselines3 PPO，支持周期断点、自动续训、学习率续接和 TensorBoard 日志。

## 目录

```text
train/
├── README.md
├── ppo_2/
│   ├── train_input_104_3二版_131rad.py
│   └── train_mohu_2_90.py
└── ppo_3/
    ├── train_mohu_10.py
    ├── train_mohu_11.py
    ├── train_mohu_12.py
    └── 断点续训说明.md
```

| 脚本 | 用途 |
| --- | --- |
| `ppo_2/train_input_104_3二版_131rad.py` | PPO_2 长距离导航，图像 + 279 维雷达 + 低维状态输入 |
| `ppo_2/train_mohu_2_90.py` | PPO_2 导航微调，带引导偏差、稠密奖励和续训 |
| `ppo_3/train_mohu_10.py` | PPO_3 末段对准，增加障碍物与叉车轨迹的净空约束 |
| `ppo_3/train_mohu_11.py` | PPO_3 末段对准，增加随机光照 |
| `ppo_3/train_mohu_12.py` | PPO_3 末段对准，增加托盘语义遮罩和独立 Actor/Critic 特征 |

## 环境

- Python 3.12
- NVIDIA GPU 和 CUDA（脚本默认使用 `device="cuda"`）
- 依赖见项目根目录的 `requirements.txt`
- `train_mohu_12.py` 使用 OpenCV，还需要安装 `opencv-python`

```powershell
cd D:\02\cha_che\提交方案\提交方案\5
python -m pip install -r requirements.txt
python -m pip install opencv-python tensorboard
```

如果使用 CPU 训练，需要将对应脚本中的 `device="cuda"` 改为 `device="cpu"`。

## 运行

脚本中的默认路径均相对于项目根目录，请先进入项目根目录再运行。训练脚本会读取
`model/forklift_train.xml`，并可从 `robot/` 下已有模型开始微调。

```powershell
# PPO_2 长距离导航
python train/ppo_2/train_input_104_3二版_131rad.py

# PPO_2 导航微调
python train/ppo_2/train_mohu_2_90.py

# PPO_3 末段对准
python train/ppo_3/train_mohu_10.py
python train/ppo_3/train_mohu_11.py
python train/ppo_3/train_mohu_12.py
```

默认会优先加载脚本同级 `train_model/` 目录中最新的断点；没有断点时再加载
`--model` 指定的模型。首次训练需要确保模型文件存在，否则从随机初始化开始。

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--xml PATH` | MuJoCo 训练场景，默认 `model/forklift_train.xml` |
| `--model PATH` | 初始模型或最终保存路径，具体默认值因脚本而异 |
| `--n-envs N` | 并行环境数，默认 `23` |
| `--total-steps N` | 累计训练目标步数，不是本次新增步数 |
| `--resume auto` | 优先使用本脚本最新断点，否则加载 `--model` |
| `--resume ckpt` | 只使用本脚本最新断点 |
| `--resume none` | 不使用已有模型，从零训练 |
| `--resume PATH` | 从指定的 `.zip` 模型或断点继续 |
| `--ckpt-interval N` | 每多少步保存一次断点，默认 `1,000,000` |
| `--lr VALUE` | 覆盖本阶段起始学习率 |
| `--save-model PATH` | 仅 `train_mohu_12.py`：训练结束时另存模型 |

示例：

```powershell
python train/ppo_3/train_mohu_12.py --resume ckpt --total-steps 12000000 --n-envs 16
python train/ppo_3/train_mohu_12.py --resume none --total-steps 1000000 --lr 1e-4
python train/ppo_3/train_mohu_12.py --save-model robot/ppo_3_v12.zip
```

## 断点与日志

每个脚本会把断点保存在自身目录下的 `train_model/`：

```text
<脚本名>_ckpt_<累计步数>steps_<批次标签>.zip
<脚本名>_latest.zip
<脚本名>_latest.json
```

`--total-steps` 表示累计目标。例如断点位于 4,000,000 步，执行
`--total-steps 6000000` 时只会再训练 2,000,000 步。

TensorBoard 日志默认写入项目根目录的 `tensorboard_logs_straight/`：

```powershell
tensorboard --logdir tensorboard_logs_straight
```

## 注意事项

- 修改奖励、成功门限或观测结构后，旧权重只能作为微调起点，不能直接视为等价模型。
- `train_mohu_12.py` 会按旧断点结构迁移可匹配的网络权重，迁移后前几次更新会重建
  Adam 动量。
- 训练脚本中的 `argparse` 帮助和关键 docstring 已保留；其余过程性注释已精简，便于阅读。
- 模型、断点和 TensorBoard 日志通常体积较大，建议仅将代码和配置提交到普通 Git
  仓库，大文件使用外部存储或 Git LFS。
