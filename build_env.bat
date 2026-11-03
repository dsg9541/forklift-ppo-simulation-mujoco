@echo off
rem ============================================================================
rem  一键构建运行环境（Windows）
rem  用法：双击本文件，或在命令行执行  build_env.bat
rem  可选：build_env.bat 环境名        默认 forklift
rem        build_env.bat forklift cpu  强制装 CPU 版 PyTorch（无 NVIDIA 显卡时用）
rem ============================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=forklift"
set "PY_VER=3.12"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu132"
if /i "%~2"=="cpu" set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"

echo [1/5] 准备 Python %PY_VER% 环境：%ENV_NAME%
where conda >nul 2>nul
if %errorlevel%==0 (
    call conda create -y -n "%ENV_NAME%" python=%PY_VER% || goto :fail
    call conda activate "%ENV_NAME%" || goto :fail
) else (
    if exist ".venv" rmdir /s /q ".venv"
    python -m venv ".venv" || goto :fail
    call ".venv\Scripts\activate.bat" || goto :fail
)

echo [2/5] 升级 pip
python -m pip install --upgrade pip || goto :fail

echo [3/5] 安装 PyTorch（%TORCH_INDEX%）
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url %TORCH_INDEX% || goto :fail

echo [4/5] 安装其余依赖
python -m pip install -r requirements.txt || goto :fail

echo [5/5] 自检
python -c "import mujoco,gymnasium,stable_baselines3,torch,numpy;print('mujoco',mujoco.__version__);print('gymnasium',gymnasium.__version__);print('sb3',stable_baselines3.__version__);print('torch',torch.__version__);print('numpy',numpy.__version__)" || goto :fail
echo.
echo 环境就绪：%ENV_NAME%
echo 接下来：  conda activate %ENV_NAME%   然后   run.bat
exit /b 0

:fail
echo.
echo [错误] 构建失败，请看上面的输出。可参考「包列表.txt」逐项手动安装。
exit /b 1
