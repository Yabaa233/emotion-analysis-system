@echo off
echo ========================================
echo   快速构建和启动 DeepFace + PyOD API
echo ========================================
echo.

cd /d "%~dp0deepface-api"

echo [1] 停止现有容器...
docker stop deepface-api 2>nul
docker rm deepface-api 2>nul

echo [2] 构建新镜像...
docker build -t deepface-pyod-api .

echo [3] 启动容器 (端口8080)...
docker run -d -p 8080:5000 --name deepface-api deepface-pyod-api

echo [4] 等待服务启动...
timeout /t 15 /nobreak >nul

echo [5] 检查服务状态...
curl -s http://localhost:8080/health
if errorlevel 1 (
    echo.
    echo ERROR: 服务启动失败
    docker logs deepface-api
    pause
    exit /b 1
)

echo.
echo [6] 检查异常检测功能...
curl -s http://localhost:8080/anomaly/info

echo.
echo ========================================
echo   服务启动成功!
echo ========================================
echo.
echo 可用端点:
echo - 健康检查: http://localhost:8080/health
echo - 情绪分析: http://localhost:8080/analyze  
echo - 异常检测: http://localhost:8080/anomaly/info
echo - Isolation Forest: http://localhost:8080/anomaly/isolation_forest
echo.
echo 容器状态:
docker ps --filter name=deepface-api --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.
pause