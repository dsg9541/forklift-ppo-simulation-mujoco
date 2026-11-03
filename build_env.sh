#!/usr/bin/env bash
# ============================================================================
#  一键构建运行环境（Linux / macOS）
#  用法：bash build_env.sh                默认环境名 forklift
#        bash build_env.sh myenv cpu      强制装 CPU 版 PyTorch
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${1:-forklift}"
BACKEND="${2:-cuda}"
PY_VER="3.12"
if [ "$BACKEND" = "cpu" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cu132"
fi

echo "[1/5] 准备 Python ${PY_VER} 环境：${ENV_NAME}"
if command -v conda >/dev/null 2>&1; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  rm -rf ".venv"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

echo "[2/5] 升级 pip"
python -m pip install --upgrade pip

echo "[3/5] 安装 PyTorch（${TORCH_INDEX}）"
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url "$TORCH_INDEX"

echo "[4/5] 安装其余依赖"
python -m pip install -r requirements.txt

echo "[5/5] 自检"
python - <<'PY'
import mujoco, gymnasium, stable_baselines3, torch, numpy
print("mujoco", mujoco.__version__)
print("gymnasium", gymnasium.__version__)
print("sb3", stable_baselines3.__version__)
print("torch", torch.__version__)
print("numpy", numpy.__version__)
PY
echo
echo "环境就绪：${ENV_NAME}"
echo "接下来：  conda activate ${ENV_NAME}   然后   bash run.sh"
