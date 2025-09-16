# 手动异常标记功能实现总结

## ✅ 已完成的功能

### 1. HTML用户界面
- 在生理信号导入面板添加了异常标记控制区域
- 启用/禁用异常标记的切换按钮
- 实时状态显示（已启用/已禁用）
- 异常标记数量显示

### 2. JavaScript核心功能
- `manualAnomalyMarks` 数组存储异常标记数据
- `anomalyMarkingEnabled` 标志控制异常标记模式
- `toggleAnomalyMarking()` 切换异常标记模式
- `addAnomalyMark()` 添加异常标记
- `removeAnomalyMark()` 移除异常标记  
- `updateAnomalyMarkingStatus()` 更新状态显示
- `updateChartAnomalyMarks()` 更新图表显示

### 3. 图表交互功能
- 增强了chartCanvas点击事件处理
- 支持Ctrl+左键点击添加异常标记
- 支持Ctrl+Shift+左键点击移除异常标记
- 自动寻找最近的valence数据点（1秒容差）
- 保留原有的时间跳转功能

### 4. VA图表可视化
- 在`va-chart.js`中添加了异常标记数据集（索引10）
- 红色星形标记样式（pointStyle: 'star'）
- `updateAnomalyMarks()` 方法更新异常标记显示
- `clearAnomalyMarks()` 方法清除异常标记

### 5. XDF导出增强
- 扩展了`convertPhysiologicalToXDF()`函数
- 添加了`ManualAnomalyMarks`数据流
- 包含AnomalyFlag和ValenceValue两个通道
- 更新了导出成功提示信息

## 🎯 功能特性

### 操作方式
- **Ctrl + 左键点击**: 在valence数据点位置添加异常标记
- **Ctrl + Shift + 左键点击**: 移除现有异常标记
- **普通点击**: 继续原有的时间跳转功能

### 数据结构
```javascript
manualAnomalyMarks = [
  { time: 12.5, valence: -0.8 },
  { time: 45.2, valence: 0.6 }
]
```

### XDF导出格式
- 数据流名称: `ManualAnomalyMarks`
- 数据流类型: `Annotations`
- 通道1: AnomalyFlag (binary) - 异常标志
- 通道2: ValenceValue (normalized) - valence值
- 时间戳: 异常发生的精确时间

## 🔧 技术实现

### 事件绑定
- DOMContentLoaded事件中绑定异常标记按钮
- 图表点击事件增强支持Ctrl+点击检测
- 自动更新图表和状态显示

### 数据管理
- 全局数组管理异常标记数据
- 实时同步图表显示
- 支持添加/移除操作
- 集成到XDF导出流程

### 用户体验
- 清晰的状态提示
- 实时的异常标记计数
- 直观的星形标记显示
- 完整的操作反馈

## 📝 使用流程
1. 导入或分析得到valence数据
2. 点击"启用异常标记"按钮
3. 在VA图表上Ctrl+点击标记异常点
4. 导出XDF文件时自动包含异常标记数据

## ✨ 核心优势
- **无侵入性**: 不影响现有功能
- **直观操作**: Ctrl+点击简单易用
- **标准格式**: 符合XDF规范
- **完整性**: 支持添加、移除、导出全流程