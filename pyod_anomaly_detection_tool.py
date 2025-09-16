#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyOD异常检测对比工具
用于对比手动标记与多种异常检测算法的效果

作者: AI Assistant
日期: 2025-09-16
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import threading

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 检查并导入PyOD
try:
    from pyod.models.ecod import ECOD
    from pyod.models.iforest import IForest  
    from pyod.models.pca import PCA
    from pyod.models.mcd import MCD
    from pyod.models.gmm import GMM
    PYOD_AVAILABLE = True
    print("✅ PyOD库导入成功")
except ImportError as e:
    PYOD_AVAILABLE = False
    print(f"❌ PyOD库导入失败: {e}")
    print("请运行: pip install pyod")

class XDFDataProcessor:
    """XDF数据处理器"""
    
    def __init__(self):
        self.va_data = None
        self.manual_marks = None
        self.timestamps = None
        self.raw_data = None
        
    def load_xdf_file(self, file_path):
        """加载XDF文件并解析数据"""
        try:
            print(f"📂 加载文件: {os.path.basename(file_path)}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                xdf_data = json.load(f)
            
            # 重置数据
            self.va_data = None
            self.manual_marks = []
            self.timestamps = None
            self.raw_data = xdf_data
            
            # 解析各个数据流
            va_stream = None
            anomaly_stream = None
            
            for stream in xdf_data.get('streams', []):
                stream_name = stream.get('info', {}).get('name', '')
                print(f"  发现数据流: {stream_name}")
                
                if 'EmotionAnalysis_VA' in stream_name:
                    va_stream = stream
                elif 'ManualAnomalyMarks' in stream_name:
                    anomaly_stream = stream
            
            # 解析VA数据
            if va_stream:
                success = self.parse_va_data(va_stream)
                if not success:
                    return False
            else:
                print("❌ 未找到VA数据流")
                return False
            
            # 解析异常标记
            if anomaly_stream:
                self.parse_anomaly_marks(anomaly_stream)
                print(f"✅ 找到 {len(self.manual_marks)} 个手动异常标记")
            else:
                print("ℹ️ 未找到手动异常标记数据流")
                
            return True
            
        except Exception as e:
            print(f"❌ 加载XDF文件失败: {e}")
            return False
    
    def parse_va_data(self, stream):
        """解析VA数据"""
        try:
            time_series = stream.get('time_series', [])
            timestamps = stream.get('time_stamps', [])
            
            if len(time_series) == 0:
                print("❌ VA数据流为空")
                return False
            
            print(f"  VA数据点: {len(time_series)}")
            
            # 转换为numpy数组
            data = np.array(time_series)
            timestamps = np.array(timestamps)
            
            # 处理数据格式 - 假设格式：[valence, arousal, face_detected]
            if data.shape[1] >= 2:
                self.va_data = data[:, :2]  # 只取valence和arousal
                self.timestamps = timestamps
                
                # 过滤无效数据（NaN或无人脸的数据点）
                if data.shape[1] >= 3:
                    face_detected = data[:, 2] > 0.5
                    valid_mask = face_detected & np.isfinite(self.va_data[:, 0]) & np.isfinite(self.va_data[:, 1])
                else:
                    valid_mask = np.isfinite(self.va_data[:, 0]) & np.isfinite(self.va_data[:, 1])
                
                self.va_data = self.va_data[valid_mask]
                self.timestamps = self.timestamps[valid_mask]
                
                print(f"  有效VA数据点: {len(self.va_data)}")
                print(f"  时间范围: {self.timestamps[0]:.2f} - {self.timestamps[-1]:.2f}秒")
                
                return True
            else:
                print(f"❌ VA数据格式不正确，期望至少2列，实际{data.shape[1]}列")
                return False
                
        except Exception as e:
            print(f"❌ 解析VA数据失败: {e}")
            return False
    
    def parse_anomaly_marks(self, stream):
        """解析手动异常标记"""
        try:
            time_series = stream.get('time_series', [])
            timestamps = stream.get('time_stamps', [])
            
            # 找到标记为异常的时间点
            anomaly_times = []
            for i, mark in enumerate(time_series):
                if len(mark) > 0 and mark[0] == 1:  # 异常标记
                    anomaly_times.append(timestamps[i])
            
            self.manual_marks = np.array(anomaly_times)
            print(f"  解析到 {len(self.manual_marks)} 个异常标记")
            
        except Exception as e:
            print(f"❌ 解析异常标记失败: {e}")
            self.manual_marks = np.array([])
    
    def extract_features(self):
        """特征工程"""
        if self.va_data is None or len(self.va_data) == 0:
            print("❌ 没有可用的VA数据进行特征提取")
            return None
            
        try:
            features = []
            
            # 原始VA值
            valence = self.va_data[:, 0]
            arousal = self.va_data[:, 1]
            features.extend([valence, arousal])
            
            # 移动平均（窗口=5）
            window = 5
            if len(valence) >= window:
                v_ma = np.convolve(valence, np.ones(window)/window, mode='same')
                a_ma = np.convolve(arousal, np.ones(window)/window, mode='same')
                features.extend([v_ma, a_ma])
            
            # 变化率
            v_diff = np.gradient(valence)
            a_diff = np.gradient(arousal)
            features.extend([v_diff, a_diff])
            
            # 幅度（距离原点的距离）
            magnitude = np.sqrt(valence**2 + arousal**2)
            features.append(magnitude)
            
            # 变化幅度
            magnitude_diff = np.gradient(magnitude)
            features.append(magnitude_diff)
            
            # 组合特征矩阵
            feature_matrix = np.column_stack(features)
            
            # 检查并处理无穷值和NaN
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"✅ 特征提取完成: {feature_matrix.shape[0]}行 x {feature_matrix.shape[1]}列")
            return feature_matrix
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return None

class AnomalyDetectionSuite:
    """异常检测算法套件"""
    
    def __init__(self, contamination=0.1):
        if not PYOD_AVAILABLE:
            self.algorithms = {}
            return
            
        # 使用适当的contamination参数来确保能检测到异常
        self.contamination = contamination
        self.algorithms = {
            'ECOD': ECOD(contamination=contamination),
            'IsolationForest': IForest(contamination=contamination, n_estimators=100, random_state=42),
            'PCA': PCA(contamination=contamination, random_state=42), 
            'MCD': MCD(contamination=contamination, random_state=42),
            'GMM': GMM(contamination=contamination, n_components=2, random_state=42)
        }
        self.results = {}
        print(f"✅ 初始化了 {len(self.algorithms)} 个异常检测算法（contamination={contamination}）")
    
    def run_all_algorithms(self, features):
        """运行所有异常检测算法，参考batch_test_all_algorithms.py的方式"""
        if not PYOD_AVAILABLE:
            print("❌ PyOD不可用，无法运行异常检测")
            return {}
            
        results = {}
        
        print("\n🔍 开始运行异常检测算法...")
        
        # 标准化数据（重要！）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        for name, algorithm in self.algorithms.items():
            print(f"  运行 {name}...")
            start_time = time.time()
            
            try:
                # 训练模型
                algorithm.fit(X_scaled)
                
                # 预测异常点
                predictions = algorithm.predict(X_scaled)  # 0: normal, 1: anomaly
                anomaly_scores = algorithm.decision_function(X_scaled)  # 异常分数
                
                execution_time = time.time() - start_time
                
                # 找到异常点的索引
                anomaly_indices = np.where(predictions == 1)[0]
                
                results[name] = {
                    'predictions': predictions,
                    'scores': anomaly_scores,
                    'anomaly_indices': anomaly_indices,
                    'execution_time': execution_time,
                    'total_anomalies_detected': len(anomaly_indices)
                }
                
                print(f"    ✅ 完成：检测到 {len(anomaly_indices)} 个异常点，耗时 {execution_time:.3f}秒")
                
            except Exception as e:
                print(f"    ❌ 失败：{e}")
                results[name] = {
                    'predictions': np.zeros(len(features)),
                    'scores': np.zeros(len(features)),
                    'anomaly_indices': [],
                    'execution_time': 0,
                    'total_anomalies_detected': 0,
                    'error': str(e)
                }
        
        self.results = results
        return results

class VisualizationGenerator:
    """可视化生成器 - 生成类似batch_test_all_algorithms.py的图表"""
    
    def __init__(self):
        self.plt_available = False
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('TkAgg')  # 使用TkAgg后端
            self.plt = plt
            self.plt_available = True
            
            # 设置中文字体
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
        except ImportError:
            print("⚠️ matplotlib未安装，将跳过可视化")
    
    def plot_single_algorithm_result(self, algorithm_name, va_data, timestamps, 
                                   predictions, scores, manual_marks, 
                                   detected_anomalies, hit_rate):
        """绘制单个算法的检测结果 - 按照参考图片格式"""
        if not self.plt_available:
            return None
            
        try:
            fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'{algorithm_name} 异常检测结果对比', fontsize=16, fontweight='bold')
            
            # 上图：VA数据时间序列（类似GSR数据）
            # 只绘制Valence数据作为主要信号（类似GSR）
            ax1.plot(timestamps, va_data[:, 0], 'b-', alpha=0.8, linewidth=1.5, label='Valence')
            
            # 标记手动异常点 - 绿色方块（参考图片风格）
            if len(manual_marks) > 0:
                manual_y_positions = []
                for mark_time in manual_marks:
                    # 找到最接近的时间点索引
                    closest_idx = np.argmin(np.abs(timestamps - mark_time))
                    if closest_idx < len(va_data):
                        y_pos = va_data[closest_idx, 0]  # Valence值
                        manual_y_positions.append(y_pos)
                        ax1.scatter([mark_time], [y_pos], 
                                  c='green', s=100, marker='s', 
                                  label='手动标注异常点' if mark_time == manual_marks[0] else "", 
                                  zorder=10, edgecolors='darkgreen', linewidth=1)
            
            # 标记检测到的异常点 - 红色圆点（参考图片风格）
            if len(detected_anomalies) > 0:
                valid_anomalies = [idx for idx in detected_anomalies if 0 <= idx < len(timestamps)]
                if valid_anomalies:
                    detected_times = timestamps[valid_anomalies]
                    detected_y_positions = va_data[valid_anomalies, 0]  # Valence值
                    ax1.scatter(detected_times, detected_y_positions, 
                              c='red', s=60, marker='o', alpha=0.8,
                              label='算法检测异常点', zorder=8, edgecolors='darkred', linewidth=1)
            
            ax1.set_xlabel('Time Index', fontsize=12)
            ax1.set_ylabel('VA (Valence)', fontsize=12)
            ax1.set_title(f'VA数据: 手动标注 vs {algorithm_name}检测结果对比\n命中情况: {hit_rate:.1f}%', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 添加统计信息文本框（参考图片风格）- 修正显示
            hit_count = int(round(hit_rate * len(manual_marks) / 100))
            info_text = f'命中情况: {hit_count}/{len(manual_marks)} = {hit_rate:.1f}%'
            ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=10, verticalalignment='top')
            
            # 下图：异常分数时间序列（参考图片的COPOD Anomaly Scores风格）
            ax2.plot(timestamps, scores, color='purple', alpha=0.7, linewidth=1.5, label='异常分数')
            
            # 在异常分数图上标记手动异常点 - 绿色方块
            if len(manual_marks) > 0:
                for mark_time in manual_marks:
                    closest_idx = np.argmin(np.abs(timestamps - mark_time))
                    if closest_idx < len(scores):
                        y_pos = scores[closest_idx]
                        ax2.scatter([mark_time], [y_pos], 
                                  c='green', s=100, marker='s', 
                                  label='手动标注的异常分数' if mark_time == manual_marks[0] else "", 
                                  zorder=10, edgecolors='darkgreen', linewidth=1)
            
            # 标记算法检测的异常分数 - 红色圆点
            if len(detected_anomalies) > 0:
                valid_anomalies = [idx for idx in detected_anomalies if 0 <= idx < len(timestamps) and 0 <= idx < len(scores)]
                if valid_anomalies:
                    detected_times = timestamps[valid_anomalies]
                    valid_scores = scores[valid_anomalies]
                    ax2.scatter(detected_times, valid_scores, 
                               c='red', s=60, marker='o', alpha=0.8,
                               label='检测异常分数', zorder=8, edgecolors='darkred', linewidth=1)
            
            ax2.set_xlabel('Time Index', fontsize=12)
            ax2.set_ylabel('Anomaly Score', fontsize=12)
            ax2.set_title(f'{algorithm_name} 异常分数', fontsize=14)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # 添加图例信息（参考图片右侧的图例风格）
            legend_elements = [
                self.plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                               markersize=8, label=f'手动标注异常点 ({len(manual_marks)})'),
                self.plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                               markersize=6, label=f'算法检测异常点 ({len(detected_anomalies)})'),
                self.plt.Line2D([0], [0], color='blue', linewidth=2, label='Valence数据'),
                self.plt.Line2D([0], [0], color='purple', linewidth=2, label='异常分数')
            ]
            
            # 在图的右侧添加统一图例
            fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=10)
            
            self.plt.tight_layout()
            self.plt.subplots_adjust(right=0.85)  # 为右侧图例留空间
            return fig
            
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")
            return None
    
    def plot_algorithms_comparison(self, all_results, manual_marks_count):
        """绘制所有算法对比图表"""
        if not self.plt_available:
            return None
            
        try:
            # 提取算法性能数据
            algorithms = []
            hit_rates = []
            execution_times = []
            detected_counts = []
            
            for name, result in all_results.items():
                if 'error' not in result:
                    algorithms.append(name)
                    hit_rates.append(result.get('hit_rate', 0))
                    execution_times.append(result.get('execution_time', 0))
                    detected_counts.append(result.get('detected_anomalies', 0))
            
            if not algorithms:
                return None
            
            # 创建2x2子图
            fig, ((ax1, ax2), (ax3, ax4)) = self.plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('PyOD算法综合性能对比', fontsize=16, fontweight='bold')
            
            colors = self.plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
            
            # 1. 命中率对比
            bars1 = ax1.barh(algorithms, hit_rates, color=colors, alpha=0.8)
            ax1.set_title('命中率对比 (%)', fontweight='bold')
            ax1.set_xlabel('命中率 (%)')
            ax1.grid(True, alpha=0.3)
            
            for i, (bar, value) in enumerate(zip(bars1, hit_rates)):
                ax1.text(value + 1, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f}%', ha='left', va='center', fontweight='bold')
            
            # 2. 检测异常点数对比
            bars2 = ax2.barh(algorithms, detected_counts, color=colors, alpha=0.8)
            ax2.set_title('检测异常点数', fontweight='bold')
            ax2.set_xlabel('检测数量')
            ax2.grid(True, alpha=0.3)
            
            for i, (bar, value) in enumerate(zip(bars2, detected_counts)):
                ax2.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                        str(int(value)), ha='left', va='center', fontweight='bold')
            
            # 3. 执行时间对比
            bars3 = ax3.barh(algorithms, execution_times, color=colors, alpha=0.8)
            ax3.set_title('执行时间对比 (秒)', fontweight='bold')
            ax3.set_xlabel('时间 (秒)')
            ax3.grid(True, alpha=0.3)
            
            for i, (bar, value) in enumerate(zip(bars3, execution_times)):
                ax3.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}s', ha='left', va='center', fontweight='bold')
            
            # 4. 算法排名总结
            ax4.axis('off')
            ranking_text = f"📊 性能总结 (手动标记: {manual_marks_count}个)\n\n"
            
            # 按命中率排序
            sorted_indices = np.argsort(hit_rates)[::-1]
            ranking_text += "🏆 命中率排名:\n"
            for i, idx in enumerate(sorted_indices[:5]):
                emoji = ["🥇", "🥈", "🥉", "🏅", "🏅"][i]
                ranking_text += f"{emoji} {algorithms[idx]}: {hit_rates[idx]:.1f}%\n"
            
            ranking_text += "\n⚡ 速度排名:\n"
            speed_indices = np.argsort(execution_times)
            for i, idx in enumerate(speed_indices[:5]):
                emoji = ["🚀", "⚡", "🏃", "🚶", "🚶"][i]
                ranking_text += f"{emoji} {algorithms[idx]}: {execution_times[idx]:.3f}s\n"
            
            ax4.text(0.1, 0.9, ranking_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace')
            
            self.plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"⚠️ 对比图绘制失败: {e}")
            return None


class ReportGenerator:
    """报告生成器 - 生成类似图片中的报告格式"""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_individual_report(self, algorithm_name, predictions, scores, manual_marks, 
                                 timestamps, execution_time, hit_rate, precision, recall, f1):
        """生成单个算法的详细报告（类似图1格式）"""
        
        report = f"""
{'='*80}
{algorithm_name} 异常检测详细报告
{'='*80}

📊 检测概要:
- 算法名称: {algorithm_name}
- 总数据点: {len(predictions)}
- 检测到异常点: {np.sum(predictions)}
- 手动标记异常点: {len(manual_marks)}
- 执行时间: {execution_time:.3f}秒

📈 性能指标:
- 命中率 (Hit Rate): {hit_rate:.1f}% ({int(hit_rate * len(manual_marks) / 100)}/{len(manual_marks)})
- 精确率 (Precision): {precision:.3f}
- 召回率 (Recall): {recall:.3f}
- F1分数: {f1:.3f}

🎯 异常点详情:
"""
        
        # 找到检测到的异常点
        try:
            if predictions is not None and len(predictions) > 0:
                anomaly_indices = np.where(predictions == 1)[0]
            else:
                anomaly_indices = []
        except Exception as e:
            print(f"⚠️ 处理预测结果时出错: {e}")
            anomaly_indices = []
        
        if len(anomaly_indices) > 0:
            report += "检测到的异常点时间戳:\n"
            for i, idx in enumerate(anomaly_indices):
                timestamp = timestamps[idx] if idx < len(timestamps) else 0
                score = scores[idx] if idx < len(scores) else 0
                
                # 检查是否命中手动标记
                hit = "✅" if self.is_hit(timestamp, manual_marks) else "❌"
                
                report += f"  {i+1:3d}. 时间: {timestamp:8.2f}s, 异常分数: {score:.3f} {hit}\n"
                
                if i >= 20:  # 限制显示前20个
                    report += f"  ... 还有 {len(anomaly_indices) - 20} 个异常点\n"
                    break
        else:
            report += "未检测到异常点\n"
        
        report += f"\n{'='*80}\n"
        return report
    
    def generate_comparison_report(self, all_results, manual_marks):
        """生成所有算法对比报告（类似图2格式）"""
        
        report = f"""
{'='*80}
合并数据集PyOD算法性能对比 (总标注异常: {len(manual_marks)}个, 容差: ±1秒)
{'='*80}

算法检测召回率:
"""
        
        # 收集所有算法的结果
        algorithm_stats = []
        
        for name, result in all_results.items():
            hit_rate = result.get('hit_rate', 0)
            execution_time = result.get('execution_time', 0)
            detected_count = result.get('detected_anomalies', 0)
            
            algorithm_stats.append({
                'name': name,
                'hit_rate': hit_rate,
                'execution_time': execution_time,
                'detected_count': detected_count,
                'hit_count': int(hit_rate * len(manual_marks) / 100)
            })
        
        # 按命中率排序
        algorithm_stats.sort(key=lambda x: x['hit_rate'], reverse=True)
        
        # 生成召回率报告
        for i, stat in enumerate(algorithm_stats):
            report += f"{stat['name']:<12} {stat['hit_count']}/{len(manual_marks)}\n"
        
        report += f"\n算法训练时间:\n"
        
        # 生成训练时间报告
        for stat in algorithm_stats:
            if stat['execution_time'] < 0.001:
                time_str = f"{stat['execution_time']*1000:.1f}ms"
            elif stat['execution_time'] < 1:
                time_str = f"{stat['execution_time']*1000:.0f}ms"
            else:
                time_str = f"{stat['execution_time']:.2f}s"
            
            report += f"{stat['name']:<12} {time_str}\n"
        
        report += f"\n{'='*80}\n"
        report += "📈 性能排名 (按命中率):\n"
        
        for i, stat in enumerate(algorithm_stats):
            emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
            report += f"{emoji} {i+1}. {stat['name']}: {stat['hit_rate']:.1f}% ({stat['hit_count']}/{len(manual_marks)})\n"
        
        report += f"\n⚡ 速度排名 (按执行时间):\n"
        
        # 按速度排序
        speed_stats = sorted(algorithm_stats, key=lambda x: x['execution_time'])
        for i, stat in enumerate(speed_stats):
            emoji = ["🚀", "⚡", "🏃"][i] if i < 3 else "🚶"
            if stat['execution_time'] < 0.001:
                time_str = f"{stat['execution_time']*1000:.1f}ms"
            elif stat['execution_time'] < 1:
                time_str = f"{stat['execution_time']*1000:.0f}ms"
            else:
                time_str = f"{stat['execution_time']:.2f}s"
            report += f"{emoji} {i+1}. {stat['name']}: {time_str}\n"
        
        report += f"\n{'='*80}\n"
        return report
    
    def is_hit(self, detected_time, manual_marks, tolerance=1.0):
        """检查检测点是否命中手动标记（±1秒容差）"""
        if len(manual_marks) == 0:
            return False
        
        distances = np.abs(manual_marks - detected_time)
        return np.any(distances <= tolerance)


class ComparisonAnalyzer:
    """结果对比分析器"""
    
    def __init__(self):
        pass
    
    def calculate_hit_rate(self, predicted_anomaly_indices, manual_mark_times, timestamps, tolerance=1.0):
        """计算命中率"""
        if len(manual_mark_times) == 0:
            return 0.0, 0, 0, []
        
        # 获取预测异常的时间戳
        predicted_times = []
        for idx in predicted_anomaly_indices:
            if idx < len(timestamps):
                predicted_times.append(timestamps[idx])
        
        hits = 0
        hit_details = []
        
        for mark_time in manual_mark_times:
            # 在容差范围内查找预测的异常点
            found_hit = False
            closest_time = None
            closest_diff = float('inf')
            
            for idx in predicted_anomaly_indices:
                if idx < len(timestamps):
                    pred_time = timestamps[idx]
                    time_diff = abs(pred_time - mark_time)
                    
                    # 记录最接近的预测时间
                    if time_diff < closest_diff:
                        closest_diff = time_diff
                        closest_time = pred_time
                    
                    if time_diff <= tolerance:
                        hits += 1
                        hit_details.append({
                            'manual_time': mark_time,
                            'predicted_time': pred_time,
                            'time_diff': time_diff
                        })
                        found_hit = True
                        break
            
            if not found_hit:
                hit_details.append({
                    'manual_time': mark_time,
                    'predicted_time': None,
                    'time_diff': None
                })
                if closest_time:
                    pass  # 可选：记录最接近的预测时间日志
        
        hit_rate = (hits / len(manual_mark_times)) * 100
        
        return hit_rate, hits, len(manual_mark_times), hit_details
    
    def generate_comparison_report(self, detection_results, manual_marks, timestamps):
        """生成对比报告"""
        report = {
            'summary': {
                'total_manual_marks': len(manual_marks),
                'timestamp_range': f"{timestamps[0]:.2f} - {timestamps[-1]:.2f}s" if len(timestamps) > 0 else "N/A",
                'data_points': len(timestamps)
            },
            'algorithms': {}
        }
        
        for alg_name, result in detection_results.items():
            if result is None or 'error' in result:
                continue
                
            # 找到预测为异常的点
            anomaly_indices = np.where(result['predictions'] == 1)[0]
            
            # 计算命中率
            hit_rate, hits, total_marks, hit_details = self.calculate_hit_rate(
                anomaly_indices, manual_marks, timestamps
            )
            
            # 计算其他指标
            total_predictions = len(anomaly_indices)
            precision = hits / total_predictions if total_predictions > 0 else 0
            recall = hit_rate  # 等同于命中率
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report['algorithms'][alg_name] = {
                'execution_time': result['execution_time'],
                'total_anomalies_detected': total_predictions,
                'manual_marks_hit': hits,
                'hit_rate': hit_rate,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'hit_details': hit_details,
                # 保存原始检测结果用于可视化
                'predictions': result['predictions'],
                'scores': result['scores'],
                'anomaly_indices': anomaly_indices
            }
        
        return report

class PyODAnomalyGUI:
    """图形用户界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PyOD异常检测对比工具 v1.0")
        self.root.geometry("1400x900")
        
        # 设置图标和样式
        try:
            self.root.iconbitmap(default="")  # 可以添加图标文件
        except:
            pass
        
        self.xdf_files = []
        self.processor = XDFDataProcessor()
        self.detector = AnomalyDetectionSuite()
        self.analyzer = ComparisonAnalyzer()
        self.visualizer = VisualizationGenerator()  # 新增可视化器
        self.all_reports = []
        
        # 检查PyOD可用性
        if not PYOD_AVAILABLE:
            messagebox.showerror("错误", "PyOD库未安装！\n\n请运行: pip install pyod")
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        # 创建样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题
        title_label = ttk.Label(main_frame, text="PyOD异常检测对比工具", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="📁 XDF文件选择")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_buttons_frame, text="选择XDF文件", command=self.select_files).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_buttons_frame, text="清空列表", command=self.clear_files).pack(side=tk.LEFT)
        
        self.file_label = ttk.Label(file_frame, text="未选择文件", foreground="gray")
        self.file_label.pack(anchor=tk.W, padx=5, pady=(0, 5))
        
        # 参数设置区域
        param_frame = ttk.LabelFrame(main_frame, text="⚙️ 参数设置")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        param_inner_frame = ttk.Frame(param_frame)
        param_inner_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_inner_frame, text="容差时间(秒):").pack(side=tk.LEFT)
        self.tolerance_var = tk.DoubleVar(value=1.0)
        tolerance_spinbox = ttk.Spinbox(param_inner_frame, from_=0.1, to=5.0, increment=0.1, 
                                       textvariable=self.tolerance_var, width=10)
        tolerance_spinbox.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(param_inner_frame, text="(使用算法默认参数)", foreground="gray").pack(side=tk.LEFT)
        
        # 控制区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 左侧按钮
        left_buttons = ttk.Frame(control_frame)
        left_buttons.pack(side=tk.LEFT)
        
        self.run_button = ttk.Button(left_buttons, text="🚀 运行异常检测", command=self.run_detection_threaded)
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(left_buttons, text="📊 导出报告", command=self.export_report).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(left_buttons, text="🔄 重置", command=self.reset_all).pack(side=tk.LEFT)
        
        # 右侧进度条
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X)
        
        self.status_label = ttk.Label(progress_frame, text="就绪", foreground="green")
        self.status_label.pack(anchor=tk.W)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="📈 检测结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建notebook用于显示不同结果
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 报告标签页
        self.create_report_tab()
        
        # 可视化标签页
        self.create_visualization_tab()
        
        # 详细结果标签页
        self.create_details_tab()
    
    def create_report_tab(self):
        """创建报告标签页"""
        self.report_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.report_frame, text="📋 对比报告")
        
        # 创建文本显示区域
        text_frame = ttk.Frame(self.report_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 文本框和滚动条
        self.report_text = tk.Text(text_frame, wrap=tk.WORD, font=('Courier New', 10))
        scrollbar_y = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        scrollbar_x = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.report_text.xview)
        
        self.report_text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 布局
        self.report_text.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
    
    def create_visualization_tab(self):
        """创建可视化标签页"""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="📊 可视化结果")
        
        # 占位文本
        placeholder_label = ttk.Label(self.viz_frame, text="运行检测后将显示可视化结果", 
                                     font=('Arial', 12), foreground="gray")
        placeholder_label.pack(expand=True)
    
    def create_details_tab(self):
        """创建详细结果标签页"""
        self.details_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.details_frame, text="🔍 详细结果")
        
        # 创建树形视图显示详细结果
        tree_frame = ttk.Frame(self.details_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 定义列
        columns = ('文件', '算法', '执行时间', '检测异常', '命中数', '命中率', 'F1分数')
        self.details_tree = ttk.Treeview(tree_frame, columns=columns, show='tree headings')
        
        # 设置列标题和宽度
        self.details_tree.heading('#0', text='项目')
        self.details_tree.column('#0', width=100)
        
        for col in columns:
            self.details_tree.heading(col, text=col)
            self.details_tree.column(col, width=100)
        
        # 滚动条
        tree_scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.details_tree.yview)
        tree_scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.details_tree.xview)
        
        self.details_tree.configure(yscrollcommand=tree_scrollbar_y.set, xscrollcommand=tree_scrollbar_x.set)
        
        # 布局
        self.details_tree.grid(row=0, column=0, sticky="nsew")
        tree_scrollbar_y.grid(row=0, column=1, sticky="ns")
        tree_scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
    
    def select_files(self):
        """选择XDF文件"""
        files = filedialog.askopenfilenames(
            title="选择XDF文件",
            filetypes=[("XDF files", "*.xdf"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if files:
            self.xdf_files = list(files)
            file_names = [os.path.basename(f) for f in files]
            if len(file_names) <= 3:
                display_text = ", ".join(file_names)
            else:
                display_text = f"{', '.join(file_names[:3])} 等 {len(file_names)} 个文件"
            
            self.file_label.config(text=f"已选择: {display_text}", foreground="blue")
            self.status_label.config(text=f"已选择 {len(files)} 个文件", foreground="blue")
    
    def clear_files(self):
        """清空文件列表"""
        self.xdf_files = []
        self.file_label.config(text="未选择文件", foreground="gray")
        self.status_label.config(text="文件列表已清空", foreground="orange")
    
    def reset_all(self):
        """重置所有内容"""
        self.clear_files()
        self.report_text.delete(1.0, tk.END)
        self.all_reports = []
        
        # 清空详细结果树
        for item in self.details_tree.get_children():
            self.details_tree.delete(item)
        
        # 清空可视化
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        placeholder_label = ttk.Label(self.viz_frame, text="运行检测后将显示可视化结果", 
                                     font=('Arial', 12), foreground="gray")
        placeholder_label.pack(expand=True)
        
        self.status_label.config(text="已重置", foreground="green")
    
    def run_detection_threaded(self):
        """在线程中运行检测，避免界面冻结"""
        if not self.xdf_files:
            messagebox.showwarning("警告", "请先选择XDF文件")
            return
        
        if not PYOD_AVAILABLE:
            messagebox.showerror("错误", "PyOD库不可用，请先安装PyOD")
            return
        
        # 禁用运行按钮
        self.run_button.config(state='disabled')
        
        # 在新线程中运行检测
        detection_thread = threading.Thread(target=self.run_detection)
        detection_thread.daemon = True
        detection_thread.start()
    
    def run_detection(self):
        """运行异常检测 - 合并所有文件为一个数据集"""
        try:
            # 更新界面状态
            self.root.after(0, lambda: self.progress.start())
            self.root.after(0, lambda: self.status_label.config(text="正在合并数据集...", foreground="orange"))
            
            # 清空之前的结果
            self.root.after(0, lambda: self.report_text.delete(1.0, tk.END))
            self.all_reports = []
            
            # 合并数据集
            combined_features = []
            combined_timestamps = []
            combined_manual_marks = []
            all_file_info = []
            
            # 添加标题到报告
            header = f"{'='*80}\n"
            header += f"合并数据集异常检测 - {len(self.xdf_files)} 个文件\n"
            header += f"{'='*80}\n\n"
            self.root.after(0, lambda text=header: self.append_to_report(text))
            
            # 处理每个文件并合并数据
            time_offset = 0.0  # 时间偏移量，确保每个文件的时间不重叠
            
            for i, file_path in enumerate(self.xdf_files):
                self.root.after(0, lambda i=i: self.update_progress_text(f"加载文件 {i+1}/{len(self.xdf_files)}..."))
                
                file_info = f"📁 文件 {i+1}: {os.path.basename(file_path)}\n"
                self.root.after(0, lambda text=file_info: self.append_to_report(text))
                
                # 加载数据
                if not self.processor.load_xdf_file(file_path):
                    error_msg = f"   ❌ 加载失败\n\n"
                    self.root.after(0, lambda text=error_msg: self.append_to_report(text))
                    continue
                
                # 提取特征
                features = self.processor.extract_features()
                if features is None:
                    error_msg = f"   ❌ 特征提取失败\n\n"
                    self.root.after(0, lambda text=error_msg: self.append_to_report(text))
                    continue
                
                # 调整时间戳（加上偏移量避免重叠）
                adjusted_timestamps = self.processor.timestamps + time_offset
                adjusted_manual_marks = [mark + time_offset for mark in self.processor.manual_marks]
                
                # 显示文件信息
                info_text = f"   ✅ 数据点: {len(features)}, 特征维度: {features.shape[1]}\n"
                info_text += f"   📍 手动标记: {len(self.processor.manual_marks)}个\n"
                info_text += f"   ⏱️  时间范围: {adjusted_timestamps[0]:.2f} - {adjusted_timestamps[-1]:.2f}秒\n\n"
                self.root.after(0, lambda text=info_text: self.append_to_report(text))
                
                # 合并到总数据集
                if len(combined_features) == 0:
                    combined_features = features
                    combined_timestamps = adjusted_timestamps
                else:
                    combined_features = np.vstack([combined_features, features])
                    combined_timestamps = np.concatenate([combined_timestamps, adjusted_timestamps])
                
                combined_manual_marks.extend(adjusted_manual_marks)
                
                # 保存文件信息
                all_file_info.append({
                    'file': os.path.basename(file_path),
                    'file_path': file_path,
                    'data_points': len(features),
                    'manual_marks': len(self.processor.manual_marks),
                    'time_range': (adjusted_timestamps[0], adjusted_timestamps[-1]),
                    'time_offset': time_offset
                })
                
                # 更新时间偏移量（下一个文件的起始时间）
                time_offset = adjusted_timestamps[-1] + 1.0  # 加1秒间隔
            
            if len(combined_features) == 0:
                error_msg = "❌ 没有成功加载任何数据文件\n"
                self.root.after(0, lambda text=error_msg: self.append_to_report(text))
                return
            
            # 显示合并后的数据集信息
            self.root.after(0, lambda: self.status_label.config(text="正在运行检测...", foreground="orange"))
            
            combined_info = f"🔗 合并数据集信息:\n"
            combined_info += f"   总数据点: {len(combined_features)}\n"
            combined_info += f"   特征维度: {combined_features.shape[1]}\n"
            combined_info += f"   总手动标记: {len(combined_manual_marks)}个\n"
            combined_info += f"   总时间范围: {combined_timestamps[0]:.2f} - {combined_timestamps[-1]:.2f}秒\n\n"
            self.root.after(0, lambda text=combined_info: self.append_to_report(text))
            
            # 计算contamination比例
            contamination = min(0.5, max(0.01, len(combined_manual_marks) / len(combined_features)))
            contamination_info = f"📊 算法参数: contamination = {contamination:.3f} ({len(combined_manual_marks)}/{len(combined_features)})\n\n"
            self.root.after(0, lambda text=contamination_info: self.append_to_report(text))
            
            # 初始化检测器
            self.detector = AnomalyDetectionSuite(contamination=contamination)
            
            # 运行异常检测
            self.root.after(0, lambda: self.update_progress_text("正在运行异常检测算法..."))
            detection_results = self.detector.run_all_algorithms(combined_features)
            
            # 生成报告
            tolerance = self.tolerance_var.get()
            report = self.analyzer.generate_comparison_report(
                detection_results, 
                combined_manual_marks, 
                combined_timestamps
            )
            
            # 添加合并信息到报告
            report['tolerance'] = tolerance
            report['file_info'] = all_file_info
            report['combined_dataset'] = True
            
            # 保存结果
            self.all_reports.append({
                'file': f"合并数据集 ({len(self.xdf_files)}个文件)",
                'file_path': "combined_dataset",
                'report': report,
                'features': combined_features,
                'detection_results': detection_results,
                'timestamps': combined_timestamps,
                'manual_marks': combined_manual_marks,
                'file_info': all_file_info
            })
            
            # 显示报告
            self.display_single_report(report)
            
            # 更新详细结果树
            self.root.after(0, self.update_details_tree)
            
            # 创建可视化
            self.root.after(0, self.create_visualizations)
            
            # 生成横向对比报告（按照参考图片格式）
            self.root.after(0, self.generate_horizontal_comparison_report)
            
        except Exception as e:
            error_msg = f"检测过程中出现错误: {e}"
            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            print(f"❌ 检测失败: {e}")
        finally:
            # 恢复界面状态
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_label.config(text="检测完成", foreground="green"))
            self.root.after(0, lambda: self.run_button.config(state='normal'))
    
    def update_progress_text(self, text):
        """更新进度文本"""
        self.status_label.config(text=text, foreground="orange")
    
    def append_to_report(self, text):
        """向报告添加文本"""
        self.report_text.insert(tk.END, text)
        self.report_text.see(tk.END)
        self.report_text.update()
    
    def display_single_report(self, report):
        """显示单个文件的报告 - 使用新的报告格式和可视化"""
        def update_ui():
            # 获取报告生成器
            report_generator = ReportGenerator()
            
            # 获取当前报告对应的数据（从all_reports获取正确的数据）
            current_report_data = None
            if self.all_reports:
                current_report_data = self.all_reports[-1]  # 最新的报告数据
            
            # 生成每个算法的详细报告和可视化
            for algo_name, result in report['algorithms'].items():
                if 'error' in result:
                    continue
                    
                # 获取需要的数据
                if 'predictions' not in result or 'scores' not in result:
                    # 如果没有原始数据，跳过可视化
                    self.report_text.insert(tk.END, f"\n{algo_name} 算法结果（无详细数据）:\n")
                    self.report_text.insert(tk.END, f"  命中率: {result.get('hit_rate', 0):.1f}%\n")
                    self.report_text.insert(tk.END, f"  执行时间: {result.get('execution_time', 0):.3f}秒\n\n")
                    continue
                
                predictions = result['predictions']
                scores = result['scores'] 
                
                # 使用合并后的数据而不是单个文件的数据
                if current_report_data:
                    manual_marks = current_report_data['manual_marks']
                    timestamps = current_report_data['timestamps']
                    va_data = current_report_data['features'][:, :2]  # 只取VA维度
                else:
                    # 后备选项
                    manual_marks = self.processor.manual_marks
                    timestamps = self.processor.timestamps
                    va_data = self.processor.va_data
                
                execution_time = result['execution_time']
                hit_rate = result['hit_rate']
                precision = result['precision']
                recall = result['recall']
                f1 = result['f1_score']
                detected_anomalies = result.get('anomaly_indices', [])
                
                # 过滤有效的异常索引，确保不超出数据范围
                valid_anomalies = [idx for idx in detected_anomalies if 0 <= idx < len(timestamps)]
                
                # 生成个体报告
                individual_report = report_generator.generate_individual_report(
                    algo_name, predictions, scores, manual_marks, timestamps,
                    execution_time, hit_rate, precision, recall, f1
                )
                
                # 添加到UI
                self.report_text.insert(tk.END, individual_report)
                
                # 生成可视化图表 - 使用过滤后的索引
                if algo_name == "IsolationForest":
                    # 为IsolationForest生成特殊的可视化
                    isolationforest_viz = IsolationForestVisualizer()
                    fig = isolationforest_viz.create_visualization(
                        va_data, timestamps, predictions, scores,
                        manual_marks, valid_anomalies, hit_rate
                    )
                else:
                    fig = self.visualizer.plot_single_algorithm_result(
                        algo_name, va_data, timestamps, predictions, scores,
                        manual_marks, valid_anomalies, hit_rate
                    )
                
                if fig:
                    # 在GUI中显示图表（可以保存为图片或嵌入）
                    import os
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    fig_path = os.path.join(temp_dir, f"{algo_name}_result.png")
                    fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)  # 使用plt.close()而不是fig.close()
                    
                    # 在报告中添加图片路径信息
                    self.report_text.insert(tk.END, f"📊 可视化图表已保存: {fig_path}\n\n")
            
            self.report_text.see(tk.END)
            self.report_text.update()
        
        self.root.after(0, update_ui)

    def display_summary_report(self):
        """显示汇总报告 - 使用新的对比报告格式和可视化"""
        def update_ui():
            # 获取报告生成器
            report_generator = ReportGenerator()
            
            # 合并所有算法结果
            combined_results = {}
            total_manual_marks = 0
            
            for report_data in self.all_reports:
                report = report_data['report']
                manual_marks = self.processor.manual_marks if hasattr(self, 'processor') else []
                total_manual_marks += len(manual_marks)
                
                for algo_name, result in report['algorithms'].items():
                    if algo_name not in combined_results:
                        combined_results[algo_name] = {
                            'hit_rate': 0,
                            'execution_time': 0,
                            'detected_anomalies': 0,
                            'count': 0
                        }
                    
                    combined_results[algo_name]['hit_rate'] += result['hit_rate']
                    combined_results[algo_name]['execution_time'] += result['execution_time'] 
                    combined_results[algo_name]['detected_anomalies'] += result.get('total_anomalies_detected', 0)
                    combined_results[algo_name]['count'] += 1
            
            # 计算平均值
            for algo_name in combined_results:
                count = combined_results[algo_name]['count']
                combined_results[algo_name]['hit_rate'] /= count
                combined_results[algo_name]['execution_time'] /= count
                combined_results[algo_name]['detected_anomalies'] = int(combined_results[algo_name]['detected_anomalies'] / count)
            
            # 生成对比报告
            avg_manual_marks = total_manual_marks // len(self.all_reports) if self.all_reports else 0
            comparison_report = report_generator.generate_comparison_report(
                combined_results, 
                list(range(avg_manual_marks))  # 估算平均手动标记数
            )
            
            # 添加到UI
            self.report_text.insert(tk.END, comparison_report)
            
            # 生成对比可视化
            fig = self.visualizer.plot_algorithms_comparison(combined_results, avg_manual_marks)
            
            if fig:
                # 保存对比图表
                import os
                import tempfile
                temp_dir = tempfile.gettempdir()
                fig_path = os.path.join(temp_dir, "algorithms_comparison.png")
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                fig.close()
                
                # 在报告中添加图片路径信息
                self.report_text.insert(tk.END, f"\n📊 算法对比图表已保存: {fig_path}\n")
                
                # 尝试显示图片（如果支持）
                try:
                    import os
                    os.startfile(fig_path)  # Windows
                except:
                    try:
                        import subprocess
                        subprocess.run(['open', fig_path])  # macOS
                    except:
                        try:
                            subprocess.run(['xdg-open', fig_path])  # Linux
                        except:
                            pass
            
            self.report_text.see(tk.END)
            self.report_text.update()
        
        self.root.after(0, update_ui)
    
    def update_details_tree(self):
        """更新详细结果树"""
        # 清空现有内容
        for item in self.details_tree.get_children():
            self.details_tree.delete(item)
        
        # 添加数据
        for i, report_data in enumerate(self.all_reports):
            file_name = report_data['file']
            report = report_data['report']
            
            # 插入文件节点
            file_item = self.details_tree.insert('', 'end', text=f"文件 {i+1}", 
                                                 values=(file_name, '', '', '', '', '', ''))
            
            # 插入算法结果
            for alg_name, metrics in report['algorithms'].items():
                self.details_tree.insert(file_item, 'end', text='', 
                                        values=('', alg_name, 
                                               f"{metrics['execution_time']:.3f}",
                                               f"{metrics['total_anomalies_detected']}",
                                               f"{metrics['manual_marks_hit']}",
                                               f"{metrics['hit_rate']:.3f}",
                                               f"{metrics['f1_score']:.3f}"))
        
        # 展开所有节点
        for item in self.details_tree.get_children():
            self.details_tree.item(item, open=True)
    
    def create_visualizations(self):
        """创建可视化图表"""
        # 清空现有内容
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        if not self.all_reports:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('PyOD异常检测结果分析', fontsize=16, fontweight='bold')
        
        # 1. 算法性能对比（命中率）
        self.plot_hit_rate_comparison(axes[0, 0])
        
        # 2. 执行时间对比
        self.plot_execution_time_comparison(axes[0, 1])
        
        # 3. F1分数对比
        self.plot_f1_score_comparison(axes[1, 0])
        
        # 4. 综合性能雷达图
        self.plot_performance_radar(axes[1, 1])
        
        plt.tight_layout()
        
        # 将图表嵌入到tkinter中
        canvas = FigureCanvasTkAgg(fig, self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_horizontal_comparison_report(self):
        """生成严格按照参考图片格式的横向对比报告"""
        if not self.all_reports:
            return
        
        try:
            # 从报告中提取数据
            current_report = self.all_reports[-1]['report']
            algorithms_data = current_report['algorithms']
            total_manual_marks = current_report['summary']['total_manual_marks']
            
            # 算法名称映射
            algo_name_mapping = {
                'ECOD': 'ECOD',
                'IsolationForest': 'IForest', 
                'PCA': 'PCA',
                'MCD': 'MCD',
                'GMM': 'GMM'
            }
            
            # 提取性能数据
            algorithms = []
            hit_rates = []
            execution_times = []
            hit_counts = []
            
            for algo_name, metrics in algorithms_data.items():
                if 'error' not in metrics:
                    display_name = algo_name_mapping.get(algo_name, algo_name)
                    algorithms.append(display_name)
                    hit_rates.append(metrics['hit_rate'])
                    execution_times.append(metrics['execution_time'])
                    
                    # 计算命中数量
                    hit_count = int(round(metrics['hit_rate'] * total_manual_marks / 100))
                    hit_counts.append(hit_count)
            
            if not algorithms:
                return
            
            # 按命中率排序（从低到高）
            sorted_indices = np.argsort(hit_rates)
            algorithms = [algorithms[i] for i in sorted_indices]
            hit_rates = [hit_rates[i] for i in sorted_indices]
            execution_times = [execution_times[i] for i in sorted_indices]
            hit_counts = [hit_counts[i] for i in sorted_indices]
            
            # 创建横向对比图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'合并数据集PyOD算法性能对比 (总标注异常: {total_manual_marks}个, 容差: ±1.0s)', 
                         fontsize=16, fontweight='bold')
            
            # 左图：异常检测召回率
            y_pos = np.arange(len(algorithms))
            bars1 = ax1.barh(y_pos, hit_rates, color='steelblue', alpha=0.8, height=0.6)
            
            # 在每个柱子上添加数值标签
            for i, (bar, hit_count) in enumerate(zip(bars1, hit_counts)):
                width = bar.get_width()
                ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                        f'{hit_count}/{total_manual_marks}', ha='left', va='center', 
                        fontsize=10, fontweight='bold')
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(algorithms, fontsize=11)
            ax1.set_xlabel('召回率 (%)', fontsize=12, fontweight='bold')
            ax1.set_title('异常检测召回率', fontsize=14, fontweight='bold')
            ax1.set_xlim(0, 100)
            ax1.grid(True, axis='x', alpha=0.3)
            ax1.set_axisbelow(True)
            
            # 添加网格线
            for i in range(0, 101, 20):
                ax1.axvline(x=i, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            
            # 右图：算法训练时间（对数刻度）
            bars2 = ax2.barh(y_pos, execution_times, color='chocolate', alpha=0.8, height=0.6)
            
            # 在每个柱子上添加时间标签
            for bar, exec_time in zip(bars2, execution_times):
                width = bar.get_width()
                if exec_time < 0.01:
                    time_label = f'{exec_time*1000:.1f}ms'
                else:
                    time_label = f'{exec_time:.2f}s'
                ax2.text(width * 1.1, bar.get_y() + bar.get_height()/2, 
                        time_label, ha='left', va='center', fontsize=10, fontweight='bold')
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(algorithms, fontsize=11)
            ax2.set_xlabel('训练时间 (秒)', fontsize=12, fontweight='bold')
            ax2.set_title('算法训练时间', fontsize=14, fontweight='bold')
            ax2.set_xscale('log')  # 对数刻度
            ax2.grid(True, axis='x', alpha=0.3)
            ax2.set_axisbelow(True)
            ax2.set_xlim(0.001, max(execution_times) * 2)
            
            plt.tight_layout()
            
            # 保存横向对比报告
            import tempfile
            temp_dir = tempfile.gettempdir()
            report_path = os.path.join(temp_dir, "PyOD_Horizontal_Comparison_Report.png")
            fig.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='white')
            
            # 关闭图表释放内存
            plt.close(fig)
            
            # 在报告文本区域添加信息
            report_info = f"\n📊 横向对比报告已生成: {report_path}\n"
            report_info += "="*60 + "\n"
            report_info += f"{'算法':<12} {'命中率':<8} {'命中数':<8} {'执行时间':<10}\n"
            report_info += "-"*60 + "\n"
            
            for i, algo in enumerate(algorithms):
                hit_rate = hit_rates[i]
                hit_count = hit_counts[i]
                exec_time = execution_times[i]
                time_str = f"{exec_time:.3f}s" if exec_time >= 0.001 else f"{exec_time*1000:.1f}ms"
                report_info += f"{algo:<12} {hit_rate:>6.1f}% {hit_count:>3}/{total_manual_marks:<3} {time_str:>8}\n"
            
            report_info += "="*60 + "\n\n"
            
            self.report_text.insert(tk.END, report_info)
            self.report_text.see(tk.END)
            
            print(f"📊 横向对比报告已保存: {report_path}")
            
        except Exception as e:
            print(f"⚠️ 生成横向对比报告失败: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_hit_rate_comparison(self, ax):
        """绘制命中率对比图"""
        if not self.all_reports:
            return
        
        # 收集数据
        algorithms = set()
        for report_data in self.all_reports:
            algorithms.update(report_data['report']['algorithms'].keys())
        
        algorithms = list(algorithms)
        hit_rates = {alg: [] for alg in algorithms}
        
        for report_data in self.all_reports:
            for alg in algorithms:
                if alg in report_data['report']['algorithms']:
                    hit_rates[alg].append(report_data['report']['algorithms'][alg]['hit_rate'])
                else:
                    hit_rates[alg].append(0)
        
        # 绘制箱线图
        data_to_plot = [hit_rates[alg] for alg in algorithms]
        bp = ax.boxplot(data_to_plot, labels=algorithms, patch_artist=True)
        
        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('命中率对比')
        ax.set_ylabel('命中率')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def plot_execution_time_comparison(self, ax):
        """绘制执行时间对比图"""
        if not self.all_reports:
            return
        
        # 计算平均执行时间
        algorithms = set()
        for report_data in self.all_reports:
            algorithms.update(report_data['report']['algorithms'].keys())
        
        algorithms = list(algorithms)
        avg_times = []
        
        for alg in algorithms:
            times = []
            for report_data in self.all_reports:
                if alg in report_data['report']['algorithms']:
                    times.append(report_data['report']['algorithms'][alg]['execution_time'])
            avg_times.append(np.mean(times) if times else 0)
        
        # 绘制条形图
        bars = ax.bar(algorithms, avg_times, color=plt.cm.Set2(np.linspace(0, 1, len(algorithms))))
        
        ax.set_title('平均执行时间对比')
        ax.set_ylabel('执行时间 (秒)')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 添加数值标签
        for bar, time in zip(bars, avg_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{time:.3f}s', ha='center', va='bottom', fontsize=8)
    
    def plot_f1_score_comparison(self, ax):
        """绘制F1分数对比图"""
        if not self.all_reports:
            return
        
        # 收集数据
        algorithms = set()
        for report_data in self.all_reports:
            algorithms.update(report_data['report']['algorithms'].keys())
        
        algorithms = list(algorithms)
        f1_scores = {alg: [] for alg in algorithms}
        
        for report_data in self.all_reports:
            for alg in algorithms:
                if alg in report_data['report']['algorithms']:
                    f1_scores[alg].append(report_data['report']['algorithms'][alg]['f1_score'])
                else:
                    f1_scores[alg].append(0)
        
        # 绘制小提琴图
        data_to_plot = [f1_scores[alg] for alg in algorithms]
        vp = ax.violinplot(data_to_plot, positions=range(len(algorithms)), showmeans=True)
        
        ax.set_title('F1分数分布')
        ax.set_ylabel('F1分数')
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45)
        ax.grid(True, alpha=0.3)
    
    def plot_performance_radar(self, ax):
        """绘制性能雷达图"""
        if not self.all_reports:
            return
        
        # 计算每个算法的平均性能指标
        algorithms = set()
        for report_data in self.all_reports:
            algorithms.update(report_data['report']['algorithms'].keys())
        
        algorithms = list(algorithms)
        
        # 性能指标：命中率、精确率、F1分数、速度（1/执行时间）
        metrics = ['命中率', '精确率', 'F1分数', '速度']
        
        # 计算每个算法的平均值
        perf_data = {}
        for alg in algorithms:
            hit_rates = []
            precisions = []
            f1_scores = []
            exec_times = []
            
            for report_data in self.all_reports:
                if alg in report_data['report']['algorithms']:
                    alg_metrics = report_data['report']['algorithms'][alg]
                    hit_rates.append(alg_metrics['hit_rate'])
                    precisions.append(alg_metrics['precision'])
                    f1_scores.append(alg_metrics['f1_score'])
                    exec_times.append(alg_metrics['execution_time'])
            
            if hit_rates:  # 如果有数据
                avg_hit_rate = np.mean(hit_rates)
                avg_precision = np.mean(precisions)
                avg_f1 = np.mean(f1_scores)
                avg_speed = 1 / np.mean(exec_times) if np.mean(exec_times) > 0 else 0
                
                # 归一化到0-1范围
                perf_data[alg] = [avg_hit_rate, avg_precision, avg_f1, avg_speed]
        
        if not perf_data:
            ax.text(0.5, 0.5, '无可视化数据', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 归一化速度指标
        all_speeds = [data[3] for data in perf_data.values()]
        max_speed = max(all_speeds) if all_speeds else 1
        for alg in perf_data:
            perf_data[alg][3] = perf_data[alg][3] / max_speed if max_speed > 0 else 0
        
        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        ax.clear()
        
        # 绘制每个算法
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        for i, (alg, data) in enumerate(perf_data.items()):
            values = data + data[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('综合性能雷达图')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
    
    def export_report(self):
        """导出报告"""
        content = self.report_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showwarning("警告", "没有可导出的报告")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存报告",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"PyOD异常检测对比报告\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"工具版本: v1.0\n")
                    f.write(f"处理文件数: {len(self.all_reports)}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(content)
                    
                    # 添加详细的算法参数信息
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"算法参数设置:\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"异常检测模式: 使用各算法默认参数\n")
                    f.write(f"容差时间 (tolerance): {self.tolerance_var.get()}秒\n")
                    f.write(f"随机种子 (random_state): 42\n")
                
                messagebox.showinfo("成功", f"报告已保存到: {file_path}")
                self.status_label.config(text=f"报告已导出: {os.path.basename(file_path)}", foreground="green")
            except Exception as e:
                messagebox.showerror("错误", f"保存报告失败: {e}")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()

def check_dependencies():
    """检查依赖项"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import seaborn
    except ImportError:
        missing_deps.append("seaborn")
    
    if not PYOD_AVAILABLE:
        missing_deps.append("pyod")
    
    if missing_deps:
        print("❌ 缺少以下依赖项:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print("✅ 所有依赖项已安装")
    return True

class IsolationForestVisualizer:
    """专门为IsolationForest创建可视化的类"""
    
    def __init__(self):
        self.plt_available = plt is not None
    
    def create_visualization(self, va_data, timestamps, predictions, scores, 
                           manual_marks, detected_anomalies, hit_rate):
        """为IsolationForest创建专门的可视化图表 - 按照参考图片格式"""
        if not self.plt_available:
            return None
            
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            fig.suptitle('VA数据: 手动标注 vs IsolationForest检测结果对比', fontsize=16, fontweight='bold')
            
            # 上图：VA数据时间序列（仿照GSR数据格式）
            # 绘制Valence数据作为主要信号
            ax1.plot(timestamps, va_data[:, 0], 'b-', alpha=0.8, linewidth=1.2, label='Valence')
            
            # 标记手动异常点 - 绿色方块（严格按照参考图片）
            manual_y_positions = []
            if len(manual_marks) > 0:
                for i, mark_time in enumerate(manual_marks):
                    # 找到最接近的时间点
                    time_diffs = np.abs(timestamps - mark_time)
                    closest_idx = np.argmin(time_diffs)
                    
                    if closest_idx < len(va_data):
                        y_pos = va_data[closest_idx, 0]  # Valence值
                        manual_y_positions.append(y_pos)
                        
                        # 绿色方块标记
                        ax1.scatter([mark_time], [y_pos], 
                                  c='green', s=120, marker='s', 
                                  label='手动标注异常点' if i == 0 else "", 
                                  zorder=15, edgecolors='darkgreen', linewidth=1.5, alpha=0.9)
            
            # 标记算法检测异常点 - 红色圆点
            detected_y_positions = []
            if len(detected_anomalies) > 0:
                valid_anomalies = [idx for idx in detected_anomalies if 0 <= idx < len(timestamps) and 0 <= idx < len(va_data)]
                if valid_anomalies:
                    detected_times = timestamps[valid_anomalies]
                    detected_y_vals = va_data[valid_anomalies, 0]  # Valence值
                    
                    # 红色圆点标记
                    ax1.scatter(detected_times, detected_y_vals, 
                              c='red', s=80, marker='o', alpha=0.8,
                              label='算法检测异常点', zorder=12, edgecolors='darkred', linewidth=1)
                    detected_y_positions = detected_y_vals
            
            # 设置坐标轴
            ax1.set_xlabel('Time Index', fontsize=12)
            ax1.set_ylabel('VA (Valence)', fontsize=12)
            # 计算实际命中数量
            hit_count = int(round(hit_rate * len(manual_marks) / 100))
            ax1.set_title(f'命中情况: {hit_count}/{len(manual_marks)} = {hit_rate:.1f}%', 
                         fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=11)
            
            # 下图：异常分数时间序列（仿照COPOD Anomaly Scores格式）
            # 分段绘制异常分数（参考图片显示的分段效果）
            mid_point = len(timestamps) // 2
            
            # 前半段 - 绿色
            ax2.plot(timestamps[:mid_point], scores[:mid_point], 
                    color='green', alpha=0.8, linewidth=1.5, label='Train Scores')
            
            # 后半段 - 紫色
            ax2.plot(timestamps[mid_point:], scores[mid_point:], 
                    color='purple', alpha=0.8, linewidth=1.5, label='Test Scores')
            
            # 在异常分数图上标记手动异常点 - 绿色方块
            if len(manual_marks) > 0:
                for i, mark_time in enumerate(manual_marks):
                    time_diffs = np.abs(timestamps - mark_time)
                    closest_idx = np.argmin(time_diffs)
                    
                    if closest_idx < len(scores):
                        y_pos = scores[closest_idx]
                        ax2.scatter([mark_time], [y_pos], 
                                  c='green', s=120, marker='s', 
                                  label='手动标注的异常分数' if i == 0 else "", 
                                  zorder=15, edgecolors='darkgreen', linewidth=1.5, alpha=0.9)
            
            # 标记算法检测的异常分数
            if len(detected_anomalies) > 0:
                valid_anomalies = [idx for idx in detected_anomalies if 0 <= idx < len(timestamps) and 0 <= idx < len(scores)]
                if valid_anomalies:
                    detected_times = timestamps[valid_anomalies]
                    detected_scores = scores[valid_anomalies]
                    ax2.scatter(detected_times, detected_scores, 
                               c='red', s=80, marker='o', alpha=0.8,
                               label='检测异常分数', zorder=12, edgecolors='darkred', linewidth=1)
            
            # 添加分割线（参考图片的Train/Test Split）
            split_time = timestamps[mid_point]
            ax1.axvline(x=split_time, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            ax2.axvline(x=split_time, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Train/Test Split')
            
            ax2.set_xlabel('Time Index', fontsize=12)
            ax2.set_ylabel('Anomaly Score', fontsize=12)
            ax2.set_title('IsolationForest 异常分数', fontsize=14)
            ax2.legend(loc='upper left', fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # 创建右侧图例（仿照参考图片）
            legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                          markersize=10, label=f'手动标注异常点 ({len(manual_marks)})'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=8, label=f'算法检测异常点 ({len(detected_anomalies)})'),
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Valence数据'),
                plt.Line2D([0], [0], color='green', linewidth=2, label='Train阶段分数'),
                plt.Line2D([0], [0], color='purple', linewidth=2, label='Test阶段分数'),
                plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Train/Test分割线')
            ]
            
            # 右侧图例
            fig.legend(handles=legend_elements, loc='center right', 
                      bbox_to_anchor=(0.98, 0.5), fontsize=10,
                      title='VA Data', title_fontsize=12)
            
            plt.tight_layout()
            plt.subplots_adjust(right=0.82)  # 为右侧图例预留空间
            
            return fig
            
        except Exception as e:
            print(f"⚠️ IsolationForest可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("🚀 启动PyOD异常检测对比工具...")
    print("="*50)
    
    # 检查依赖
    if not check_dependencies():
        input("按Enter键退出...")
        return
    
    try:
        app = PyODAnomalyGUI()
        app.run()
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        input("按Enter键退出...")

if __name__ == "__main__":
    main()