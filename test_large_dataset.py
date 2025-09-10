#!/usr/bin/env python3
"""
测试大数据集NETS异常检测
"""

import numpy as np
import requests
import json

def test_large_dataset_detection():
    """测试大数据集异常检测"""
    
    print("🔍 测试大数据集NETS异常检测")
    print("=" * 50)
    
    # 模拟大数据集（类似GSR情感数据）
    np.random.seed(42)
    
    # 生成17952个数据点（类似你之前遇到的情况）
    large_data = []
    
    # 基础信号：低频波动
    for i in range(17952):
        base_signal = 0.5 * np.sin(i * 0.001) + 0.3 * np.sin(i * 0.003)
        noise = np.random.normal(0, 0.1)
        large_data.append(base_signal + noise)
    
    # 在特定位置添加情感响应异常（模拟真实情况）
    emotion_events = [2000, 5500, 8200, 11000, 14500, 16800]  # 6个情感事件
    for pos in emotion_events:
        if pos < len(large_data):
            # 模拟GSR峰值
            for offset in range(-5, 6):
                if 0 <= pos + offset < len(large_data):
                    large_data[pos + offset] += 0.8 * np.exp(-abs(offset) * 0.5)
    
    print(f"📊 生成测试数据:")
    print(f"  - 数据点数: {len(large_data)}")
    print(f"  - 数据范围: [{min(large_data):.3f}, {max(large_data):.3f}]")
    print(f"  - 模拟情感事件位置: {emotion_events}")
    
    # 调用NETS API
    api_url = 'http://localhost:5001/nets/java/detect'
    
    test_cases = [
        {"name": "默认参数", "params": {}},
        {"name": "高敏感", "params": {"R": 0.03, "K": 3}},
        {"name": "平衡设置", "params": {"R": 0.05, "K": 5}},
    ]
    
    for case in test_cases:
        print(f"\n🎯 测试案例: {case['name']}")
        print("-" * 30)
        
        try:
            payload = {
                "data": large_data,
                "params": case["params"]
            }
            
            response = requests.post(api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                if result['status'] == 'success':
                    print(f"✅ 检测成功!")
                    print(f"  - 检测到异常点: {result['outlier_count']}个")
                    print(f"  - 异常率: {result['anomaly_rate']:.2f}%")
                    print(f"  - 检测用时: {result['execution_time']:.2f}s")
                    print(f"  - 参数: {result['parameters']}")
                    
                    # 检查是否检测到模拟的情感事件
                    detected_indices = result['outlier_indices']
                    detected_emotions = []
                    for event_pos in emotion_events:
                        nearby_detections = [idx for idx in detected_indices 
                                           if abs(idx - event_pos) <= 50]
                        if nearby_detections:
                            detected_emotions.append(event_pos)
                    
                    print(f"  - 检测到的情感事件: {len(detected_emotions)}/{len(emotion_events)}")
                    print(f"    成功检测: {detected_emotions}")
                    
                    if result['outlier_count'] > 10:
                        print(f"  🎉 解决了'只检测到2个异常点'的问题!")
                    else:
                        print(f"  ⚠️  异常点数量仍然较少，需要调整参数")
                        
                else:
                    print(f"❌ 检测失败: {result.get('message', '未知错误')}")
            else:
                print(f"❌ API调用失败: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")

if __name__ == '__main__':
    test_large_dataset_detection()
