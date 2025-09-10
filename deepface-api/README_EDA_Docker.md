# EDA预处理 Docker 部署指南

## 📦 Docker 环境部署

### 1. 构建和启动服务

```bash
# 进入API目录
cd c:\MyWork\RunMorphCast\deepface-api

# 构建Docker镜像
docker build -t deepface-eda-api .

# 使用docker-compose启动服务
docker-compose up -d

# 或者直接运行容器
docker run -d -p 5000:5000 --name deepface-eda-api deepface-eda-api
```

### 2. 验证部署

```bash
# 检查容器状态
docker ps

# 查看容器日志
docker logs deepface-eda-api

# 测试API健康状态
curl http://localhost:5000/health

# 测试EDA功能
curl http://localhost:5000/eda/info
```

## 🧪 EDA API 使用说明

### 1. 信号清理 API

**端点:** `POST /eda/clean`

**请求示例:**
```json
{
  "signal": [0.1, 0.12, 0.15, 0.13, 0.11, 0.14, ...],
  "sampling_rate": 50.0,
  "method": "neurokit"
}
```

**方法选项:**
- `neurokit`: NeuroKit2官方方法（推荐）
- `biosppy`: BioSPPy兼容方法
- `cvxeda`: cvxEDA分解方法
- `none`: 无预处理

**响应示例:**
```json
{
  "status": "success",
  "data": {
    "cleaned_signal": [0.11, 0.115, 0.12, ...],
    "method_used": "neurokit",
    "sampling_rate": 50.0,
    "original_length": 1000,
    "processed_length": 1000,
    "quality_score": 0.85,
    "preprocessing_info": {
      "neurokit_version": "0.2.7",
      "filters_applied": "neurokit_method"
    }
  },
  "neurokit_available": true
}
```

### 2. 信号分解 API

**端点:** `POST /eda/decompose`

**请求示例:**
```json
{
  "signal": [0.1, 0.12, 0.15, ...],
  "sampling_rate": 50.0
}
```

**响应示例:**
```json
{
  "status": "success",
  "data": {
    "tonic": [0.11, 0.111, 0.112, ...],   // 慢性成分 (SCL)
    "phasic": [0.01, 0.015, 0.008, ...],  // 快性成分 (SCR)
    "clean": [0.11, 0.115, 0.12, ...],    // 清理后信号
    "peaks": [45, 123, 267, ...],         // SCR峰值位置
    "sampling_rate": 50.0,
    "decomposition_method": "neurokit2"
  },
  "neurokit_available": true
}
```

### 3. 功能信息 API

**端点:** `GET /eda/info`

查看EDA预处理功能的可用性和配置信息。

## 🔧 前端集成

HTML文件中的JavaScript现在会：

1. **优先使用官方NeuroKit2**: 通过API调用Python服务
2. **自动降级**: API不可用时使用备用的JavaScript实现
3. **进度显示**: 显示API连接和处理状态
4. **错误处理**: 网络问题时优雅降级

### 使用流程:

1. 用户选择预处理方法（neurokit/biosppy/cvxeda/none）
2. 点击"NETS检测"按钮
3. 系统尝试连接Docker中的NeuroKit2 API
4. 如果API可用：使用官方NeuroKit2处理
5. 如果API不可用：自动使用备用JavaScript方法
6. 显示处理结果和质量评分

## 🚀 性能对比

| 方法 | 处理质量 | 速度 | 功能完整性 |
|------|----------|------|------------|
| NeuroKit2 API | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 备用JavaScript | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 📝 注意事项

1. **Docker服务**: 确保Docker容器正在运行
2. **网络连接**: 前端需要能访问 http://localhost:5000
3. **依赖安装**: requirements.txt包含所有必需的Python包
4. **内存使用**: NeuroKit2处理大量数据时需要足够内存
5. **超时设置**: API调用有30秒超时，超时后自动降级

## 🛠️ 故障排除

### 常见问题:

1. **API连接失败**: 检查Docker容器是否运行
2. **NeuroKit2导入错误**: 重新构建Docker镜像
3. **内存不足**: 增加Docker内存限制
4. **处理速度慢**: 考虑减少数据点数量或使用更快的方法

### 调试命令:

```bash
# 查看API日志
docker logs -f deepface-eda-api

# 进入容器调试
docker exec -it deepface-eda-api bash

# 重启服务
docker-compose restart
```
