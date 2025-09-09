# MorphCast + DeepFace 双API情感分析系统启动指南

## 快速启动（推荐方式）

### 1. 启动DeepFace API服务
```bash
cd c:\MyWork\RunMorphCast\deepface-api
start_docker_service.bat
```

### 2. 验证API服务
```bash
test_api.bat
```

### 3. 打开HTML应用
直接双击打开：`Test_MorphCast_Vedio_VA_prehand_fixed.html`

## 详细步骤

### 第一步：启动DeepFace API
1. 确保Docker Desktop正在运行
2. 进入deepface-api目录
3. 运行启动脚本：`start_docker_service.bat`
4. 等待看到 "DeepFace API is running on http://localhost:5000"

### 第二步：测试API连接
运行 `test_api.bat` 确保API正常响应

### 第三步：使用HTML应用
1. 打开 `Test_MorphCast_Vedio_VA_prehand_fixed.html`
2. 选择API类型：
   - "MorphCast (云端)" - 使用云端服务
   - "DeepFace (本地)" - 使用本地Docker服务
3. 开始视频分析

## 故障排除

### 如果Docker服务启动失败：
```bash
# 检查Docker状态
docker ps

# 重新构建镜像
cd deepface-api
docker build -f Dockerfile.minimal -t deepface-test .
docker run -d -p 5000:5000 --name deepface-api deepface-test
```

### 如果API无响应：
```bash
# 检查容器状态
docker ps -a

# 查看容器日志
docker logs deepface-api

# 重启容器
docker restart deepface-api
```

## 停止服务
```bash
cd deepface-api
stop_service.bat
```

## 文件说明
- `Test_MorphCast_Vedio_VA_prehand_fixed.html` - 主应用文件
- `va-chart.js` - 图表显示组件
- `deepface-api/` - DeepFace API服务目录
