# 手动异常标记功能使用指南

## 功能概述
在VA图表上手动标记valence数据的异常点，支持Ctrl+点击操作和XDF导出。

## 使用步骤

### 1. 启用异常标记模式
- 在"生理信号导入"面板中找到"异常标记"部分
- 点击"启用异常标记"按钮
- 状态显示为"✅ 异常标记已启用"

### 2. 标记异常点
- **添加异常标记**: 按住 `Ctrl` + 鼠标左键点击VA图表上的valence数据点
- **移除异常标记**: 按住 `Ctrl` + `Shift` + 鼠标左键点击已标记的异常点
- 异常点会在图表上显示为红色星形标记 ⭐

### 3. 异常标记说明
- 系统会自动找到点击位置最近的valence数据点（1秒容差范围内）
- 每个异常标记包含时间点和对应的valence值
- 异常标记会实时更新在图表上

### 4. XDF导出
- 点击"转换并导出XDF"按钮
- 导出的XDF文件将包含一个新的数据流: `ManualAnomalyMarks`
- 异常标记数据流包含:
  - AnomalyFlag: 异常标志（1=异常，0=正常）
  - ValenceValue: 异常点的valence值
  - 时间戳: 异常发生的时间点

## XDF数据结构示例
```json
{
  "info": {
    "name": "ManualAnomalyMarks",
    "type": "Annotations",
    "channel_count": 2,
    "channels": [
      {
        "label": "AnomalyFlag",
        "unit": "binary",
        "description": "Manual anomaly detection flag (1=anomaly, 0=normal)"
      },
      {
        "label": "ValenceValue", 
        "unit": "normalized",
        "description": "Valence value at anomaly point (-1 to 1)"
      }
    ]
  },
  "time_series": [[1, -0.8], [1, 0.6]], // [异常标志, valence值]
  "time_stamps": [12.5, 45.2] // 时间戳
}
```

## 注意事项
- 异常标记功能需要先有valence数据才能使用
- 导入生理信号数据或现有XDF文件都支持异常标记
- 异常标记会保存在导出的XDF文件中，便于后续分析
- 图表上的红色星形标记表示手动标记的异常点

## 技术细节
- 异常检测容差: 1秒
- 标记样式: 红色星形，大小8px
- 数据格式: XDF 1.0 标准
- 支持的操作: 添加、移除、导出