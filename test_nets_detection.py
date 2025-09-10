#!/usr/bin/env python3
"""
测试Java NETS异常点检测和索引获取
"""

import requests
import json

# 创建测试数据：包含明显异常的信号
test_data = [1.0] * 100 + [10.0, 15.0, 12.0] + [1.0] * 50 + [20.0] + [1.0] * 47

print(f"测试数据: {len(test_data)}个点")
print(f"异常点应该在索引: 100, 101, 102, 153")

# 调用API
response = requests.post('http://localhost:5001/nets/java/detect', json={
    'data': test_data,
    'params': {
        'R': 0.5,  # 严格的距离阈值
        'K': 10,   # 邻域要求
        'W': min(10000, len(test_data)),
        'S': 50,
        'D': 1,
        'sD': 1,
        'nW': 1
    }
})

if response.status_code == 200:
    result = response.json()
    print(f"\n检测结果:")
    print(f"状态: {result.get('status')}")
    print(f"异常点数量: {result.get('outlier_count', 0)}")
    print(f"异常点索引: {result.get('outlier_indices', [])}")
    print(f"异常率: {result.get('anomaly_rate', 0):.2f}%")
    print(f"执行时间: {result.get('cpu_time', 0):.3f}s")
    print(f"内存使用: {result.get('memory_usage', 0):.1f}MB")
else:
    print(f"API调用失败: {response.status_code}")
    print(f"错误信息: {response.text}")
