# 🎭 MorphCast情绪分析系统

一个基于MorphCast和DeepFace的实时视频情绪分析系统，支持VA(Valence-Arousal)模型和7种基础情绪检测，具备完整的数据处理和导出功能。

## ✨ 主要功能

### 🎯 情绪分析
- **双API支持**: MorphCast (实时) + DeepFace (高精度)
- **VA模型**: Valence-Arousal二维情绪空间分析
- **7种情绪**: 愤怒、厌恶、恐惧、快乐、悲伤、惊讶、中性
- **区域检测**: 可自定义面部检测区域
- **实时处理**: 视频播放与情绪数据同步显示

### 📊 数据处理
- **滤波处理**: 中值滤波、异常值裁剪
- **标准化**: 零均值化、归零缩放、Z-Score标准化
- **导数计算**: Savitzky-Golay、斜率、平滑导数
- **事件检测**: 自动识别情绪变化事件
- **Rate-of-Change分析**: 情绪变化率分析

### 📈 可视化
- **实时图表**: VA值和情绪概率的动态图表
- **交互播放**: 点击图表跳转到对应时间点
- **区域显示**: 可视化面部检测区域
- **数据叠加**: 多种数据处理结果的对比显示

### 💾 数据管理
- **XDF格式**: 符合LSL标准的数据导出/导入
- **完整性保证**: 包含所有时间点的数据
- **批量处理**: 支持多文件处理
- **格式兼容**: 与LabRecorder、EEGLAB等工具兼容

## 🚀 快速开始

### 环境要求
- 现代浏览器 (Chrome, Firefox, Edge)
- Docker (用于DeepFace API)
- Python 3.8+ (可选，用于独立DeepFace)

### 启动方式

#### 方法1: 一键启动 (推荐)
```bash
# Windows
START_ALL.bat

# 手动启动
docker-compose up -d
# 然后打开 Test_MorphCast_Vedio_VA_prehand_fixed.html
```

#### 方法2: 单独启动
```bash
# 启动DeepFace API
cd deepface-api
docker build -t deepface-api .
docker run -p 5000:5000 deepface-api

# 在浏览器中打开主文件
# Test_MorphCast_Vedio_VA_prehand_fixed.html
```

### 使用流程
1. **选择视频文件** - 支持常见视频格式
2. **配置分析参数** - 帧间隔、API选择、区域检测等
3. **开始分析** - 自动处理并生成情绪数据
4. **数据处理** - 使用内置工具优化数据质量
5. **导出结果** - XDF格式，与LSL生态系统兼容

## 📁 项目结构

```
RunMorphCast/
├── Test_MorphCast_Vedio_VA_prehand_fixed.html  # 主应用程序
├── va-chart.js                                 # 图表组件
├── deepface-api/                              # DeepFace Docker API
│   ├── Dockerfile
│   ├── deepface_api.py
│   ├── requirements.txt
│   └── docker-compose.yml
├── LSL_XDF_Integration_Guide.md               # XDF格式集成指南
├── START_ALL.bat                              # 一键启动脚本
├── STOP_ALL.bat                               # 停止脚本
└── START_GUIDE.md                             # 启动指南
```

## 🔧 技术栈

### 前端
- **HTML5/JavaScript**: 现代Web技术
- **Chart.js**: 数据可视化
- **MorphCast SDK**: 实时情绪检测
- **Canvas API**: 视频帧处理

### 后端
- **Python**: DeepFace API服务
- **Flask**: Web API框架
- **OpenCV**: 图像处理
- **DeepFace**: 深度学习情绪识别
- **Docker**: 容器化部署

### 数据格式
- **XDF**: LSL标准数据交换格式
- **JSON**: 配置和临时数据
- **Canvas ImageData**: 实时帧数据

## 📊 数据处理算法

### 预处理
- **中值滤波**: 去除短期噪声
- **异常值检测**: 基于统计方法的离群点处理
- **插值填充**: 处理缺失数据

### 标准化方法
- **零均值化**: 去除系统偏差，保持原始变异性
- **归零缩放**: 统一范围到[-1,1]，保持相对比例
- **Z-Score**: 标准正态分布，消除个体差异

### 导数计算
- **Savitzky-Golay**: 平滑导数，保持特征
- **简单斜率**: 快速变化率计算
- **平滑导数**: 多点平均，减少噪声

## 🎯 应用场景

### 研究领域
- **心理学研究**: 情绪反应分析
- **人机交互**: 用户体验评估
- **医疗健康**: 情感状态监测
- **教育评估**: 学习情绪分析

### 商业应用
- **广告效果**: 观众情绪反馈
- **产品测试**: 用户情感体验
- **客服质量**: 情绪满意度分析
- **娱乐内容**: 观众参与度评估

## 🔬 数据质量保证

### 检测鲁棒性
- **多API融合**: MorphCast + DeepFace双重验证
- **区域限制**: 减少背景干扰
- **置信度评估**: 基于检测质量的数据筛选

### 处理可靠性
- **时间同步**: 精确的时间戳管理
- **数据完整性**: 全时间轴数据保证
- **格式标准**: LSL兼容的XDF格式

## 📈 性能优化

### 实时处理
- **帧采样**: 可调节的处理间隔
- **异步处理**: 非阻塞API调用
- **缓存机制**: 减少重复计算

### 资源管理
- **内存优化**: 及时释放无用数据
- **Docker容器**: 隔离的API服务
- **批量处理**: 提高大文件处理效率

## 🤝 贡献指南

### 开发环境
1. Clone项目
2. 安装Docker
3. 运行`START_ALL.bat`启动服务
4. 在浏览器中测试

### 代码结构
- 前端逻辑在主HTML文件中
- 图表组件在`va-chart.js`
- API服务在`deepface-api/`目录

## 📄 许可证

本项目基于MIT许可证开源，详见LICENSE文件。

## 🆘 故障排除

### 常见问题
1. **Docker启动失败**: 检查Docker Desktop是否运行
2. **API连接错误**: 确认5000端口未被占用
3. **视频加载失败**: 检查浏览器是否支持视频格式
4. **数据导出问题**: 确认有分析数据且格式正确

### 性能建议
- 使用较低的视频分辨率
- 调整帧采样间隔
- 启用区域检测减少计算量
- 定期清理浏览器缓存

## 📞 支持

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 查看文档: `LSL_XDF_Integration_Guide.md`
- 参考启动指南: `START_GUIDE.md`

---

**享受探索人类情绪的奇妙之旅！** 🎭✨
