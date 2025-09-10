#!/usr/bin/env python3
"""
简单的API服务器，用于测试Java NETS集成
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_java_nets import get_simple_java_nets_detector

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/nets/java/check', methods=['GET'])
def check_java_nets():
    """检查Java NETS环境"""
    try:
        detector = get_simple_java_nets_detector()
        env_status = detector.check_environment()
        return jsonify(env_status)
    except Exception as e:
        return jsonify({
            'java_available': False,
            'error': str(e)
        })

@app.route('/nets/java/detect', methods=['POST'])
def detect_anomalies():
    """Java NETS异常检测端点"""
    try:
        data = request.json
        signal = data.get('data', [])
        params = data.get('params', {})
        
        if not signal or len(signal) < 10:
            return jsonify({
                'status': 'error',
                'error': 'Signal too short for NETS analysis (minimum 10 points)'
            })
        
        print(f"收到检测请求: {len(signal)}个数据点")
        print(f"参数: {params}")
        
        # 获取Java NETS检测器
        detector = get_simple_java_nets_detector()
        
        # 转换参数
        nets_params = {
            'W': params.get('W', min(1000, len(signal))),
            'S': params.get('S', 100),
            'R': params.get('R', 0.45),  # STK数据集默认值
            'K': params.get('K', 50),
            'D': params.get('D', 1),     # 单维数据
            'sD': params.get('sD', 1),
            'nW': params.get('nW', 1)
        }
        
        # 执行检测
        result = detector.detect_anomalies(signal, **nets_params)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Java NETS检测错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'ok', 'message': 'Java NETS API服务正常运行'})

if __name__ == '__main__':
    print("启动Java NETS API服务...")
    print("访问 http://localhost:5001/health 进行健康检查")
    print("访问 http://localhost:5001/nets/java/check 检查Java环境")
    app.run(host='0.0.0.0', port=5001, debug=True)
