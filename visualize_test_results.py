#!/usr/bin/env python3
"""
可视化Java NETS测试结果
"""

import numpy as np
import matplotlib.pyplot as plt
from simple_java_nets import SimpleJavaNETSDetector

def visualize_nets_detection():
    """可视化NETS异常检测结果"""
    
    # 创建检测器
    detector = SimpleJavaNETSDetector()
    
    # 生成测试数据（与API中相同的数据）
    np.random.seed(42)
    test_data = np.random.normal(0, 1, 500).tolist()
    
    # 添加几个明显的异常点
    test_data[100] += 5
    test_data[200] += 5
    test_data[300] += 5
    
    print("生成测试数据...")
    print(f"数据点数: {len(test_data)}")
    print(f"手动添加异常点位置: [100, 200, 300]")
    
    # 执行检测
    print("\n执行NETS异常检测...")
    result = detector.detect_anomalies(test_data)
    
    print(f"检测结果:")
    print(f"  检测到异常点数: {result['outlier_count']}")
    print(f"  异常率: {result['anomaly_rate']:.1f}%")
    print(f"  异常点索引: {result['outlier_indices'][:10]}{'...' if len(result['outlier_indices']) > 10 else ''}")
    
    # 创建可视化图表
    plt.figure(figsize=(15, 8))
    
    # 绘制原始数据
    x = range(len(test_data))
    plt.subplot(2, 1, 1)
    plt.plot(x, test_data, 'b-', linewidth=1, alpha=0.7, label='原始数据')
    
    # 标记手动添加的异常点
    manual_outliers = [100, 200, 300]
    for idx in manual_outliers:
        plt.plot(idx, test_data[idx], 'ro', markersize=8, label='手动异常点' if idx == manual_outliers[0] else "")
    
    plt.title('原始测试数据（包含手动异常点）')
    plt.xlabel('数据点索引')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制检测结果
    plt.subplot(2, 1, 2)
    plt.plot(x, test_data, 'b-', linewidth=1, alpha=0.5, label='原始数据')
    
    # 标记NETS检测到的异常点
    detected_outliers = result['outlier_indices']
    for idx in detected_outliers:
        if idx < len(test_data):
            plt.plot(idx, test_data[idx], 'r*', markersize=6, 
                    label='NETS检测异常点' if idx == detected_outliers[0] else "")
    
    # 高亮手动添加的异常点
    for idx in manual_outliers:
        plt.plot(idx, test_data[idx], 'go', markersize=10, markerfacecolor='none', 
                markeredgewidth=2, label='手动异常点' if idx == manual_outliers[0] else "")
    
    plt.title(f'NETS异常检测结果 (检测到 {result["outlier_count"]} 个异常点)')
    plt.xlabel('数据点索引')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = 'nets_detection_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存: {output_file}")
    
    plt.show()
    
    # 分析检测效果
    print(f"\n检测效果分析:")
    detected_manual = [idx for idx in manual_outliers if idx in detected_outliers]
    print(f"  手动异常点检测成功: {len(detected_manual)}/{len(manual_outliers)}")
    print(f"  成功检测的手动异常点: {detected_manual}")
    
    # 统计不同区域的异常点分布
    regions = {
        '前1/4 (0-124)': [idx for idx in detected_outliers if 0 <= idx < 125],
        '中间1/2 (125-374)': [idx for idx in detected_outliers if 125 <= idx < 375],
        '后1/4 (375-499)': [idx for idx in detected_outliers if 375 <= idx < 500]
    }
    
    print(f"\n异常点分布:")
    for region, indices in regions.items():
        print(f"  {region}: {len(indices)}个异常点")
    
    return result

if __name__ == '__main__':
    try:
        result = visualize_nets_detection()
    except Exception as e:
        print(f"可视化错误: {e}")
        print("请确保已安装matplotlib: pip install matplotlib")
