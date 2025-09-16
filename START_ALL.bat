@echo off
echo ========================================
echo   MorphCast + DeepFace System Startup
echo ========================================
echo.

echo [Step 1] Checking Docker Desktop...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)
echo Docker Desktop is running.

echo.
echo [Step 2] Starting DeepFace API service...
cd /d "%~dp0deepface-api"

echo Stopping existing containers...
docker stop deepface-api 2>nul
docker rm deepface-api 2>nul

echo Building and starting DeepFace + PyOD API...
docker build -f Dockerfile -t deepface-pyod-api .
docker run -d -p 8080:5000 --name deepface-api ^
    --memory=2g ^
    --cpus=2 ^
    --shm-size=1g ^
    deepface-pyod-api

echo Waiting for service startup...
timeout /t 10 /nobreak >nul

docker ps | findstr deepface-api >nul
if errorlevel 1 (
    echo ERROR: Container failed to start
    docker logs deepface-api
    pause
    exit /b 1
) else (
    echo SUCCESS: DeepFace + PyOD API is running on http://localhost:8080
)

echo.
echo [Step 3] Opening HTML application...
cd /d "%~dp0"
start "" "Test_MorphCast_Vedio_VA_prehand_fixed.html"

echo.
echo ========================================
echo   System is ready!
echo ========================================
echo.
echo Services available:
echo - Emotion Analysis: http://localhost:8080/analyze
echo - Anomaly Detection: http://localhost:8080/anomaly/info
echo - Health Check: http://localhost:8080/health
echo.
echo To stop the service, run: STOP_ALL.bat
echo.
