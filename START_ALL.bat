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

echo Building and starting DeepFace API...
docker build -f Dockerfile.minimal -t deepface-test .
docker run -d -p 5000:5000 --name deepface-api ^
    --memory=2g ^
    --cpus=2 ^
    --shm-size=1g ^
    deepface-test

echo Waiting for service startup...
timeout /t 10 /nobreak >nul

docker ps | findstr deepface-api >nul
if errorlevel 1 (
    echo ERROR: Container failed to start
    docker logs deepface-api
    pause
    exit /b 1
) else (
    echo SUCCESS: DeepFace API is running on http://localhost:5000
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
echo You can now:
echo 1. Select API type in the web page
echo 2. Start video analysis
echo.
echo To stop the service, run: STOP_ALL.bat
echo.
