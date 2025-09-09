@echo off
echo ========================================
echo   Stopping MorphCast + DeepFace System
echo ========================================
echo.

echo [Step 1] Stopping DeepFace API service...
docker stop deepface-api 2>nul
if errorlevel 1 (
    echo Container already stopped or not found
) else (
    echo Container stopped successfully
)

echo.
echo [Step 2] Removing container...
docker rm deepface-api 2>nul
if errorlevel 1 (
    echo Container already removed or not found
) else (
    echo Container removed successfully
)

echo.
echo [Step 3] Cleaning up...
docker container prune -f >nul 2>&1

echo.
echo ========================================
echo   All services stopped successfully!
echo ========================================
echo.
pause
