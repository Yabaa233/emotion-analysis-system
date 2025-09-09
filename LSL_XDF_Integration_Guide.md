# XDF 格式集成指南

## 概述
我们已经成功将 XDF 格式集成到情绪分析系统中，替换了原有的 CSV 格式。XDF 格式符合 LSL (Lab Streaming Layer) 标准，可以被标准 LSL 工具读取和处理。

## 前端 (HTML) 实现的功能

### 1. XDFFormatter 类
- **XDF 格式导出**: 将分析数据转换为符合 LSL 标准的 XDF 格式
- **XDF 格式导入**: 读取 XDF 文件并恢复完整的分析数据
- **多流支持**: VA 数据流 + 情绪数据流的独立管理
- **完整性保证**: 包含所有数据点，包括无人脸时段

### 2. 用户界面更新
- **导出按钮**: "下载 CSV" → "导出 XDF"
- **导入按钮**: "导入 CSV" → "导入 XDF"
- **保持现有功能**: 视频同步播放、图表交互、数据处理

### 3. XDF 数据流设计
- **VA 数据流**: 名称 `EmotionAnalysis_VA`，包含 Valence、Arousal、FaceDetected 3个通道
- **情绪数据流**: 名称 `EmotionAnalysis_Emotions`，包含 7 种情绪的概率值
- **时间同步**: 精确的时间戳匹配和数据对应

## 后端 (Python API) 优化

### 1. 性能优化
- **图像缩放**: 自动将大图像缩放到 640px 以提高处理速度
- **快速检测器**: 使用 OpenCV 后端替代默认检测器
- **宽松检测**: 允许低置信度的面部检测，减少处理时间

### 2. 批量处理 (新增)
- **批量端点**: `/analyze_batch` 支持多图像并行处理
- **并发处理**: 使用 ThreadPoolExecutor 提高吞吐量
- **错误处理**: 单个图像失败不影响整体批处理

## 使用流程

### 1. 标准分析流程
1. 选择视频文件
2. 点击 "Analyze" 进行预处理
3. 分析完成后自动生成 VA 和情绪数据
4. 使用现有的本地播放功能观看同步结果

### 2. 数据导出/导入
1. 分析完成后点击 "导出 XDF" 保存数据
2. 使用 "导入 XDF" 加载之前的分析结果
3. XDF 文件包含完整的 VA 值、情绪数据和人脸检测状态
4. 导入后可进行完整的数据处理和 Rate-of-Change 分析

## XDF 格式要求

### 不需要额外安装
- ✅ 无需安装 LSL JavaScript 库
- ✅ 无需安装 LabRecorder（除非需要与其他 LSL 工具集成）
- ✅ 纯 JavaScript 实现，开箱即用

## XDF 文件格式

### 数据结构
```json
{
  "info": {
    "version": "1.0",
    "created": "2024-01-01T00:00:00.000Z",
    "software": "EmotionAnalysisApp"
  },
  "streams": [
    {
      "info": {
        "name": "EmotionAnalysis_VA",
        "type": "Markers",
        "channel_count": 3,
        "channels": [
          {"label": "Valence", "unit": "normalized", "type": "VA"},
          {"label": "Arousal", "unit": "normalized", "type": "VA"},
          {"label": "FaceDetected", "unit": "boolean", "type": "Marker"}
        ]
      },
      "time_series": [[valence1, arousal1, face1], [valence2, arousal2, face2], ...],
      "time_stamps": [timestamp1, timestamp2, ...]
    },
    {
      "info": {
        "name": "EmotionAnalysis_Emotions",
        "type": "Markers", 
        "channel_count": 7,
        "channels": [
          {"label": "Angry", "unit": "probability", "type": "Emotion"},
          {"label": "Disgust", "unit": "probability", "type": "Emotion"},
          // ... 其他情绪
        ]
      },
      "time_series": [[angry1, disgust1, ...], [angry2, disgust2, ...], ...],
      "time_stamps": [timestamp1, timestamp2, ...]
    }
  ]
}
```

## 测试 LabRecorder 兼容性

### 1. 自动兼容性检查
系统在导出XDF文件时会自动进行兼容性检查：
- ✅ **格式验证**: 检查XDF结构是否符合LSL标准
- ✅ **数据完整性**: 验证通道数量、时间戳一致性
- ✅ **元数据规范**: 确认StreamInfo格式正确
- ✅ **测试报告**: 自动生成详细的兼容性报告

### 2. LabRecorder 测试步骤

#### 安装 LabRecorder
1. 下载: https://github.com/labstreaminglayer/App-LabRecorder/releases
2. 选择适合你系统的版本（Windows/Mac/Linux）
3. 解压并运行 LabRecorder

#### 测试导出的XDF文件
1. **分析视频并导出XDF**：
   - 在应用中完成视频分析
   - 点击 "导出 XDF"
   - 查看兼容性测试报告
   - 下载 `.xdf` 文件和 `.txt` 测试报告

2. **在LabRecorder中打开**：
   - 启动 LabRecorder
   - File → Open → 选择你的 `.xdf` 文件
   - 检查是否能正确加载和显示数据

3. **验证数据流**：
   - 确认能看到 `EmotionAnalysis_VA` 流（3通道）
   - 确认能看到 `EmotionAnalysis_Emotions` 流（7通道，如果启用）
   - 检查时间轴是否正确
   - 验证数据值范围是否合理

### 3. EEGLAB 测试（可选）

如果你有MATLAB和EEGLAB：

```matlab
% 在MATLAB中测试
[streams, fileheader] = load_xdf('your_emotion_file.xdf');

% 查看数据流信息
fprintf('发现 %d 个数据流\n', length(streams));
for i = 1:length(streams)
    fprintf('流 %d: %s (%d 通道)\n', i, streams{i}.info.name, streams{i}.info.channel_count);
end

% 绘制VA数据
if ~isempty(streams)
    va_stream = streams{1}; % 假设第一个是VA流
    figure;
    subplot(2,1,1);
    plot(va_stream.time_stamps, va_stream.time_series(1,:));
    title('Valence');
    ylabel('Value');
    
    subplot(2,1,2);
    plot(va_stream.time_stamps, va_stream.time_series(2,:));
    title('Arousal'); 
    ylabel('Value');
    xlabel('Time (s)');
end
```

### 4. 常见兼容性问题及解决方案

#### 问题1: LabRecorder无法打开文件
**可能原因**: XDF格式不符合标准
**解决方案**: 
- 检查兼容性测试报告中的错误信息
- 确保所有必需字段都存在
- 验证数据类型正确（数字、字符串等）

#### 问题2: 通道数量不匹配
**可能原因**: 声明的通道数与实际数据不符
**解决方案**:
- 自动兼容性检查会发现此问题
- 系统会自动修正通道数量

#### 问题3: 时间戳问题
**可能原因**: 时间戳格式或数量不正确
**解决方案**:
- 系统使用相对时间戳（从0开始）
- 确保每个数据点都有对应的时间戳

### 5. 验证数据质量

#### VA数据验证
- Valence 范围: -1 到 +1
- Arousal 范围: -1 到 +1  
- FaceDetected: 0 或 1

#### 情绪数据验证
- 每个情绪概率: 0 到 1
- 所有情绪概率之和应接近 1
- 主导情绪应对应最高概率值

### 6. 导出文件说明

导出时会生成两个文件：
1. **`emotion_analysis_YYYYMMDDTHHMMSS.xdf`**: 主数据文件
2. **`compatibility_report_YYYYMMDDTHHMMSS.txt`**: 兼容性测试报告

测试报告包含：
- 文件基本信息
- 数据流详细信息  
- 兼容性检查结果
- 测试建议和步骤

## 故障排除

### XDF 兼容性问题
- **检查兼容性报告**: 每次导出都会生成详细的测试报告
- **验证必需字段**: 确保所有LSL标准字段都存在
- **数据类型检查**: 确认数值类型正确（浮点数、整数等）

### LabRecorder 读取问题
- **文件路径**: 确保文件路径不包含特殊字符
- **文件大小**: 过大的文件可能加载缓慢
- **LSL版本**: 确保使用最新版本的LabRecorder

### 数据质量问题  
- **时间戳连续性**: 检查时间轴是否合理
- **数值范围**: 验证VA值在[-1,1]范围内
- **缺失数据**: 确认无人脸时段的处理是否正确

### 性能优化
- 使用较低的视频分辨率
- 调整帧采样间隔
- 大文件导出可能需要较长时间

## 未来扩展

### 可选的 LSL 实时流
如果需要 LSL 实时流功能，可以轻松添加：
1. 重新集成 LSL JavaScript 库
2. 在 XDFFormatter 基础上添加流管理
3. 实现 LabRecorder 实时录制

### 其他格式支持
- 可扩展支持其他神经科学数据格式
- 保持 XDF 作为标准交换格式
- 考虑添加 EDF、BrainVision 等格式
