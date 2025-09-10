#!/usr/bin/env python3

print("测试大数据集...")

# 生成17952个数据点
import random
large_data = [random.random() for _ in range(17952)]

print(f"生成了 {len(large_data)} 个数据点")
print("现在的参数设置应该会:")
print("- 使用多窗口分析 (nW > 1)")
print("- 更敏感的参数 (R=0.05, K=5)")
print("- 保持大部分原始数据结构")

# 检查参数逻辑
data_size = len(large_data)
if data_size > 10000:
    W = min(2000, data_size // 8)  # 窗口大小
    S = W // 20  # 步长
    nW = max(8, data_size // W)  # 多窗口
    R = 0.05
    K = 5
    print(f"\n计算出的参数:")
    print(f"W={W}, S={S}, nW={nW}, R={R}, K={K}")
    print(f"窗口数量: {nW} (这应该能检测到更多异常点!)")
else:
    print("数据量不足以触发大数据集模式")
