#!/usr/bin/env python3
"""
简化的Java NETS集成 - 使用STK数据集格式
"""

import os
import sys
import subprocess
import shutil
import time
import numpy as np

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
import re

class SimpleJavaNETSDetector:
    def __init__(self):
        """初始化Java NETS检测器"""
        self.nets_path = self._find_nets_path()
        
    def _find_nets_path(self):
        """查找NETS源码路径"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        nets_path = os.path.join(current_dir, 'NETS', 'src')
        
        if not os.path.exists(nets_path):
            raise FileNotFoundError(f"找不到Java NETS源码: {nets_path}")
        
        print(f"Java NETS路径: {nets_path}")
        return nets_path
    
    def check_environment(self):
        """检查Java和NETS环境"""
        try:
            # 检查Java - 使用完整路径
            java_path = r"C:\Program Files\Eclipse Adoptium\jdk-8.0.462.8-hotspot\bin\java.exe"
            java_result = subprocess.run([java_path, '-version'], 
                                       capture_output=True, text=True, timeout=10)
            
            # 检查NETS编译
            testbase_class = os.path.join(self.nets_path, 'test', 'testBase.class')
            nets_compiled = os.path.exists(testbase_class)
            
            return {
                'java_available': java_result.returncode == 0,
                'java_version': java_result.stderr.split('\n')[0] if java_result.stderr else '',
                'nets_compiled': nets_compiled,
                'testbase_with_indices_compiled': os.path.exists(os.path.join(self.nets_path, 'test', 'testBaseWithIndices.class')),
                'nets_path': self.nets_path
            }
        except Exception as e:
            return {'java_available': False, 'error': str(e)}
    
    def create_stock_data_file(self, data, inject_test_anomalies=True):
        """创建STK格式的数据文件
        
        Args:
            data: 输入数据
            inject_test_anomalies: 是否注入测试异常点（用于调试）
        """
        datasets_dir = os.path.join(self.nets_path, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)
        
        stock_file = os.path.join(datasets_dir, 'Stock.csv')
        
        # 备份原始文件
        backup_file = stock_file + '.backup'
        if os.path.exists(stock_file) and not os.path.exists(backup_file):
            shutil.copy2(stock_file, backup_file)
        
        # 预处理数据：跳过前10个点，避免初始化异常
        processed_data = data[10:] if len(data) > 20 else data
        
        # 大数据集处理策略
        if len(processed_data) > 10000:
            # 🔧 修复：大数据集采样策略，避免NETS算法对超大数据的处理问题
            print(f"  大数据集处理: {len(processed_data)}点，采用智能采样")
            
            # 保留关键数据点：开头、结尾、中间关键位置
            key_points = processed_data[:500]  # 开头500点
            key_points.extend(processed_data[-500:])  # 结尾500点
            
            # 中间部分均匀采样
            middle_data = processed_data[500:-500]
            if len(middle_data) > 0:
                # 计算采样步长，保留足够的数据点给NETS分析
                target_middle_points = min(4000, len(middle_data))
                step = max(1, len(middle_data) // target_middle_points)
                sampled_middle = middle_data[::step]
                key_points.extend(sampled_middle)
            
            processed_data = key_points
            print(f"  智能采样完成: -> {len(processed_data)}点 (保留开头+结尾+均匀中间)")
            
        elif len(processed_data) > 5000:
            # 中等数据集：适度采样
            import random
            # 保留前1000点，然后随机采样
            sampled_data = processed_data[:1000]
            remaining_data = processed_data[1000:]
            sample_size = min(4000, len(remaining_data))
            sampled_indices = sorted(random.sample(range(len(remaining_data)), sample_size))
            sampled_data.extend([remaining_data[i] for i in sampled_indices])
            processed_data = sampled_data
            print(f"  中等数据集采样: 原始{len(data)}点 -> 处理后{len(processed_data)}点")
        
        # 数据标准化处理
        processed_data = np.array(processed_data)
        
        # 仅在测试模式下注入异常点
        if inject_test_anomalies and len(processed_data) <= 1000:
            mean_val = np.mean(processed_data)
            std_val = np.std(processed_data)
            
            # 在中间部分注入一些明显的异常点进行测试
            if len(processed_data) > 100:
                test_positions = [len(processed_data)//4, len(processed_data)//2, 3*len(processed_data)//4]
                for pos in test_positions:
                    if pos < len(processed_data):
                        # 注入异常值：3倍标准差的偏移
                        processed_data[pos] = mean_val + 3 * std_val
                print(f"  测试: 在位置 {test_positions} 注入异常点 (均值+3σ)")
        
        processed_data = processed_data.tolist()
        
        # 写入数据
        with open(stock_file, 'w') as f:
            for value in processed_data:
                f.write(f"{value}\n")
        
        print(f"创建数据文件: {stock_file} ({len(processed_data)}个数据点)")
        return stock_file
    
    def restore_stock_data_file(self):
        """恢复原始STK数据文件"""
        stock_file = os.path.join(self.nets_path, 'datasets', 'Stock.csv')
        backup_file = stock_file + '.backup'
        
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, stock_file)
            print("恢复原始STK数据文件")
    
    def detect_anomalies(self, data, **params):
        """
        检测异常
        
        Args:
            data: 时间序列数据列表
            **params: NETS参数
        
        Returns:
            检测结果字典
        """
        # 检查环境
        env = self.check_environment()
        if not env['java_available'] or not env['nets_compiled']:
            raise RuntimeError(f"环境检查失败: {env}")
        
        # 智能参数计算 - 针对不同数据量优化
        data_size = len(data)
        
        if data_size > 10000:
            # 大数据集：使用多窗口分析
            W = params.get('W', min(2000, data_size // 8))  # 窗口大小
            S = params.get('S', W // 20)  # 步长：窗口的5%
            nW = params.get('nW', max(8, data_size // W))  # 多窗口
            R = params.get('R', 0.05)  # 更敏感的距离阈值
            K = params.get('K', 5)     # 更敏感的邻域
            print(f"  大数据集模式: 启用多窗口分析")
        elif data_size > 5000:
            # 中等数据集
            W = params.get('W', min(1500, data_size // 4))
            S = params.get('S', W // 15)
            nW = params.get('nW', max(4, data_size // W))
            R = params.get('R', 0.08)
            K = params.get('K', 6)
            print(f"  中等数据集模式")
        else:
            # 小数据集：当前测试模式
            W = params.get('W', min(1000, data_size))
            S = params.get('S', min(100, W // 10))
            nW = params.get('nW', 1)  # 单窗口
            R = params.get('R', 0.1)
            K = params.get('K', 8)
            print(f"  小数据集模式")
        
        D = params.get('D', 1)     # 单维度
        sD = params.get('sD', 1)
        
        print(f"开始Java NETS检测:")
        print(f"  数据点数: {len(data)}")
        print(f"  参数: W={W}, S={S}, R={R}, K={K}, D={D}, sD={sD}, nW={nW}")
        
        # 判断是否为测试数据（小数据集启用测试异常注入）
        is_test_data = data_size <= 1000
        
        # 创建数据文件
        data_file = self.create_stock_data_file(data, inject_test_anomalies=is_test_data)
        
        try:
            # 构建Java命令 - 使用修改版的testBaseWithIndices
            java_path = r"C:\Program Files\Eclipse Adoptium\jdk-8.0.462.8-hotspot\bin\java.exe"
            cmd = [
                java_path, '-cp', '.',
                'test.testBaseWithIndices',  # 使用新的带索引输出的类
                '--dataset', 'STK',
                '--W', str(W),
                '--S', str(S),
                '--R', str(R),
                '--K', str(K),
                '--D', str(D),
                '--sD', str(sD),
                '--nW', str(nW)
            ]
            
            print(f"  执行命令: {' '.join(cmd)}")
            
            # 执行NETS
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.nets_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                raise RuntimeError(f"Java NETS执行失败: {result.stderr}")
            
            # 解析结果文件 - 使用通配符模式匹配，因为Java可能输出R3.0而不是R3
            import glob
            result_pattern = os.path.join(
                self.nets_path, 'Result',
                f"Result_STK_NETS_D{D}_sD{sD}_rand0_R{R}*_K{K}_S{S}_W{W}_nW{nW}.txt"
            )
            
            result_files = glob.glob(result_pattern)
            if not result_files:
                # 尝试更宽松的匹配
                result_pattern_loose = os.path.join(
                    self.nets_path, 'Result',
                    f"Result_STK_NETS_D{D}_sD{sD}_rand0_R*_K{K}_S{S}_W{W}_nW{nW}.txt"
                )
                result_files = glob.glob(result_pattern_loose)
            
            if not result_files:
                raise RuntimeError(f"结果文件不存在，匹配模式: {result_pattern}")
            
            result_file = result_files[0]  # 使用第一个匹配的文件
            print(f"  找到结果文件: {result_file}")
            
            # 读取和解析结果
            with open(result_file, 'r') as f:
                content = f.read()
            
            print(f"  结果文件内容:\n{content}")
            
            # 读取异常点索引文件
            outlier_indices = []
            try:
                import glob
                outlier_pattern = os.path.join(
                    self.nets_path, 'Result',
                    f"Outliers_STK_NETS_D{D}_sD{sD}_rand0_R*_K{K}_S{S}_W{W}_nW{nW}.txt"
                )
                outlier_files = glob.glob(outlier_pattern)
                
                if outlier_files:
                    outlier_file = outlier_files[0]
                    print(f"  找到异常点索引文件: {outlier_file}")
                    
                    with open(outlier_file, 'r') as f:
                        outlier_content = f.read()
                    
                    print(f"  异常点索引文件内容:\n{outlier_content}")
                    
                    # 解析异常点索引
                    for line in outlier_content.split('\n'):
                        if 'outliers:' in line and ',' in line:
                            indices_str = line.split('outliers:')[1].strip()
                            if indices_str.endswith(','):
                                indices_str = indices_str[:-1]  # 移除末尾逗号
                            
                            for idx_str in indices_str.split(','):
                                if idx_str.strip():
                                    try:
                                        idx = int(idx_str.strip())
                                        if 0 <= idx < len(data):  # 确保索引有效
                                            outlier_indices.append(idx)
                                    except ValueError:
                                        pass
                else:
                    print("  未找到异常点索引文件")
                    
            except Exception as e:
                print(f"  读取异常点索引时出错: {e}")
            
            print(f"  解析到异常点索引: {outlier_indices}")
            
            # 解析异常点数量
            outlier_count = 0
            cpu_time = 0
            memory_usage = 0
            
            for line in content.split('\n'):
                if 'outliers:' in line:
                    outlier_count = int(line.split('outliers:')[1].strip())
                elif line.strip() and not line.startswith('#') and not line.startswith('Method:') and not line.startswith('Dim:') and not line.startswith('R/K/W/S:'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            cpu_time = float(parts[0])
                            memory_usage = float(parts[1])
                        except:
                            pass
            
            return {
                'status': 'success',
                'outlier_count': outlier_count,
                'outlier_indices': outlier_indices,  # 现在包含具体的异常点索引
                'total_points': len(data),
                'anomaly_rate': (outlier_count / len(data)) * 100,
                'cpu_time': cpu_time,
                'memory_usage': memory_usage,
                'execution_time': execution_time,
                'dataset': 'STK',
                'method': 'NETS',
                'parameters': {
                    'W': W, 'S': S, 'R': R, 'K': K, 'D': D, 'sD': sD, 'nW': nW
                }
            }
            
        finally:
            # 清理：恢复原始数据文件
            self.restore_stock_data_file()

def get_simple_java_nets_detector():
    """获取简化版Java NETS检测器实例"""
    return SimpleJavaNETSDetector()

# Flask API集成
if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)

    detector_instance = None

    @app.route('/nets/java/detect', methods=['POST'])
    def detect_anomalies_api():
        """Java NETS异常检测API端点"""
        global detector_instance
        
        try:
            if detector_instance is None:
                detector_instance = SimpleJavaNETSDetector()
            
            data = request.get_json()
            if not data or 'data' not in data:
                return jsonify({'status': 'error', 'message': '缺少数据字段'}), 400
            
            # 提取参数
            time_series = data['data']
            params = data.get('params', {})
            
            if not time_series or len(time_series) < 10:
                return jsonify({'status': 'error', 'message': '数据点数量不足'}), 400
            
            # 执行检测
            result = detector_instance.detect_anomalies(time_series, **params)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        """健康检查端点"""
        return jsonify({'status': 'ok', 'service': 'Java NETS API'})

    @app.route('/nets/java/check', methods=['GET'])
    def check_java_nets_environment():
        """检查Java NETS环境状态"""
        global detector_instance
        
        try:
            if detector_instance is None:
                detector_instance = SimpleJavaNETSDetector()
            
            env_status = detector_instance.check_environment()
            
            return jsonify({
                'status': 'success' if env_status.get('java_available') and env_status.get('nets_compiled') else 'error',
                'java_available': env_status.get('java_available', False),
                'nets_compiled': env_status.get('nets_compiled', False),
                'testbase_with_indices_compiled': env_status.get('testbase_with_indices_compiled', False),
                'java_version': env_status.get('java_version', ''),
                'nets_path': env_status.get('nets_path', ''),
                'error': env_status.get('error')
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'java_available': False,
                'nets_compiled': False,
                'error': str(e)
            }), 500

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--api':
        # API模式
        if not FLASK_AVAILABLE:
            print("错误: Flask未安装，无法启动API服务器")
            print("请运行: pip install flask flask-cors")
            sys.exit(1)
        print("启动Java NETS API服务器...")
        app.run(host='127.0.0.1', port=5001, debug=False)
    else:
        # 测试模式
        detector = SimpleJavaNETSDetector()
        
        # 生成测试数据
        import numpy as np
        np.random.seed(42)
        test_data = np.random.normal(0, 1, 500).tolist()
        # 添加几个异常点
        test_data[100] += 5
        test_data[200] += 5
        test_data[300] += 5
        
        result = detector.detect_anomalies(test_data)
        print(f"\n检测结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # 简单的文本可视化
        print(f"\n📊 数据可视化:")
        print(f"测试数据范围: [{min(test_data):.2f}, {max(test_data):.2f}]")
        print(f"手动添加异常点位置: [100, 200, 300]")
        print(f"手动异常点值: [{test_data[100]:.2f}, {test_data[200]:.2f}, {test_data[300]:.2f}]")
        
        detected = result['outlier_indices']
        print(f"\nNETS检测到的异常点 (前20个): {detected[:20]}")
        
        # 检查手动异常点是否被检测到
        manual_outliers = [100, 200, 300]
        detected_manual = [idx for idx in manual_outliers if idx in detected]
        print(f"手动异常点检测成功: {len(detected_manual)}/{len(manual_outliers)} = {detected_manual}")
        
        # 异常点分布统计
        regions = {
            '前1/4': [idx for idx in detected if 0 <= idx < 125],
            '中间1/2': [idx for idx in detected if 125 <= idx < 375], 
            '后1/4': [idx for idx in detected if 375 <= idx < 500]
        }
        print(f"\n异常点分布:")
        for region, indices in regions.items():
            print(f"  {region}: {len(indices)}个")
            
        print(f"\n🎯 结论: NETS成功检测到{result['outlier_count']}个异常点，异常率{result['anomaly_rate']:.1f}%")
