# 具身智能平衡重叉车 · MuJoCo + PPO

> **版本**：46 号 = 冠军 `v3_fwd046`（2026-09-25 归档）
> **仿真引擎**：MuJoCo 3.12.0（CPU 物理 + OpenGL 离屏渲染），Python 3.12
> **任务**：在仓库仿真环境中完成平衡重叉车的长距离导航、货叉对准、取货与放货。
> **方法**：Stable-Baselines3 PPO，融合门架相机、279 维雷达和低维状态。

公开仓库包含推理代码、训练脚本、场景数据、工具和模型权重。

---

## 一、目录

```
├── main.py                 主程序（冠军 test_5_warehouse.py 原样，仅改名 + 启动选点界面）
├── f_c/                    核心库（与冠军逐字节一致）
│   └── train/              策略网络定义（原 train_*.py）
├── train/                  训练脚本与训练说明
├── model/                  场景与数据（与冠军逐字节一致）
├── robot/                  PPO 模型权重（Git LFS）
├── tools/                  9 个复现与重标定脚本
├── 妙妙工具/                场景与货堆编辑器（原样带上，见第六节）
├── docs/                   46 号报告 + 提交包说明（等价性 / 默认三轮的实测筛选）
├── README.md / requirements.txt / _variant.json / eval_score.json
├── .gitignore / .gitattributes
└── build_env.bat / build_env.sh    一键建环境脚本
```

## 二、安装

```powershell
git clone https://github.com/dsg9541/forklift-ppo-simulation-mujoco.git
cd forklift-ppo-simulation-mujoco
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

模型权重通过 Git LFS 保存。若没有执行 `git lfs pull`，`robot/` 下只会得到 LFS
指针文件，无法直接运行完整流程。

训练脚本说明见 [`train/README.md`](train/README.md)。

## 三、运行

```powershell
set PYTHONUTF8=1                  :: 建议先设，控制台按 UTF-8 输出（见第七节）

python main.py                    :: 弹窗选三轮放货目标 → 跑三轮
python main.py --no-target-ui     :: 不弹窗，直接用默认三轮（A2 / B2 / A3，演示种子 2200）
python main.py --demo-targets A2,B2,A3 --seed 0   :: 逐字复现冠军的那一组
python main.py --no-render        :: 无窗口复跑
```

| 参数 | 作用 |
| --- | --- |
| `--no-target-ui` | 不弹选点界面，用默认三轮 |
| `--demo-targets A2,B2,A3` | 直接指定三轮放货目标（不弹界面） |
| `--seed N` | 随机种子（默认 2200，演示种子）；种子决定每轮的引导降质采样 |
| `--no-render` | 不开可视化窗口（渲染与判定照常） |
| `--no-camera` | 关门架相机渲染（仅排查用） |
| `--no-model` | 不加载 PPO，只跑底层流程自检 |
| `--no-guidance` | 关闭引导降质 |
| `--group N` | 用 `model/eval_groups.json` 第 N 组任务替代默认三轮 |

## 四、实测数据（46 号冠军自身 496 轮，见 `docs/46_报告.md`）

| 指标 | 数值 |
| --- | --- |
| 总轮数 / 成功轮数 | 496 / 485（97.8%） |
| 成功判据 | 无碰撞，且落点 < 10 厘米、角度 < 5 度 |
| 成功轮落点误差 | 中位 3.58 厘米，均值 3.84，90 分位 6.38，最差 9.84 |
| 成功轮角度误差 | 中位 0.80 度，均值 0.94，90 分位 2.00，最差 4.30 |
| 成功轮每轮物理步 | 中位 110,499（约 110.5 秒），均值 116,470（约 116.5 秒） |
| 组均物理步（3 轮一组） | 346,935（约 346.9 秒） |
| 失败轮 | 11 轮（2.2%）：6 轮碰撞、5 轮近失 |

> 时间口径：1 物理步 = 1 毫秒仿真时间。

默认演示三轮（**seed 2200**，A2 / B2 / A3）实测：三轮全成功、零碰撞，累计 351,578 物理步
（约 351.6 秒），三轮落点 1.81 / 4.74 / 4.43 厘米。第 1 轮（要录屏的那一轮）比 seed 0
更快也更干净：112,861 步、卡住 3 次、复核 2 次、**不用倒退重进站**（seed 0 是 134,828 步、
卡住 5 次、复核 6 次、重进站 1 次）。

## 五、模型权重校验

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `robot/ppo_2.zip` | 长距离导航 | `49EE59D37B5B6EAFBC0806D74D559BDF2EC55E7B04297D3501D8744C01B9468D` |
| `robot/ppo_3_v3.zip` | 末段视觉对准（实际生效） | `63B08A915207AB551EC61D4B4E3F7F1ED98AE3F41BAD05E276E682D417FEF10E` |

策略网络结构定义在 `f_c/train/` 下（`train_input_104_3二版_131rad.py`、`train_mohu_12.py`），
部署加载权重时会读取它们——网络结构、权重、训练脚本三件齐全。

下载后可以校验权重：

```powershell
Get-FileHash robot\ppo_2.zip -Algorithm SHA256
Get-FileHash robot\ppo_3_v3.zip -Algorithm SHA256
```

## 六、tools 保留清单（9 个）

| 脚本 | 作用 |
| --- | --- |
| `validate_scene.py` | 场景与数据一致性检查 |
| `calibrate_heights.py` | 重新生成取放高度标定表 |
| `insert_probe.py` | 插叉几何与插入深度标定 |
| `dyn_id.py` | 二自由度车辆模型 (κ, β) 辨识 |
| `turn_probe.py` | 偏航率与转向执行器实测 |
| `ppo3_probe.py` | ppo_3 对准收敛包线标定 |
| `dock_probe.py` | 相机横向 / 雷达法向位姿扫描 |
| `geom_probe.py` | 足迹净空几何检查 |
| `check_camera.py` | 确认门架相机挂在叉车子树里 |

## 七、妙妙工具（原样带上）

`妙妙工具/` 是配套的场景与货堆编辑工具，与冠军目录里的那一份**逐字节一致、未做任何改动**：

```
妙妙工具/
├── robot_map_editor/      仓库地图编辑器（Tkinter 画布：货堆增删、拖拽、导出 XML / JSON）
├── warehouse/27/          编辑器用的 27 号仓库工程（warehouse_env.xml / warehouse_info.json / path_graph.json）
├── build_27_scene.py      由工程文件生成 27 号场景
└── test_stack_3layers.xml 三层货堆单测场景
```

## 八、说明

- 引导位姿按赛方规定的精度降质：位置幅值 0–0.2 米均匀随机、姿态 ±10° 均匀随机，每轮取货点与放货点各采样一次，
  轮内保持不变；采样只由 `--seed` 决定，同种子可复现。
- 默认三轮目标仍是冠军自带的 A2 / B2 / A3（取放高度是冠军写死的，第 1 轮要放两层垛、
  第 2/3 轮放一层垛，所以候选本来就有限）；默认**种子改成 2200**，因为第 1 轮的取货腿
  只由种子决定——seed 2200 是记录里用过的演示种子，第 1 轮又快又干净。全部实测对照见
  `docs/提交包说明_与冠军等价性.md` 第 4 节。
- Windows 控制台请先 `set PYTHONUTF8=1`（或 `chcp 65001`）：程序日志里有 `−`、`→` 等符号，
  默认 GBK 控制台会在打印时抛 `UnicodeEncodeError`。这条与物理、判定无关，只影响日志输出。
