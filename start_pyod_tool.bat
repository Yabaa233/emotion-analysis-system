@echo off
echo ========================================
echo PyOD异常检测工具启动器
echo ========================================
echo.

:: 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo Python已安装，版本信息：
python --version
echo.

:: 检查是否已安装依赖
echo 检查依赖包...
python -c "import numpy, pandas, matplotlib, seaborn, pyod" >nul 2>&1
if %errorlevel% neq 0 (
    echo 检测到缺少依赖包，是否自动安装？
    set /p install_deps="输入 y 安装依赖，或 n 跳过: "
    if /i "%install_deps%"=="y" (
        echo 正在安装依赖...
        python install_dependencies.py
        echo.
    )
)

:: 启动主程序
echo 启动PyOD异常检测工具...
echo.
python pyod_anomaly_detection_tool.py

if %errorlevel% neq 0 (
    echo.
    echo 程序运行出错，请检查错误信息
)

echo.
echo 程序已结束
pause