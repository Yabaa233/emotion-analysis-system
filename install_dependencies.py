#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyOD异常检测工具依赖安装脚本
"""

import subprocess
import sys
import os

def install_package(package):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 安装失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    """主安装函数"""
    print("🚀 PyOD异常检测工具依赖安装程序")
    print("="*50)
    
    # 需要安装的包列表
    required_packages = [
        "numpy",
        "pandas", 
        "matplotlib",
        "seaborn",
        "scikit-learn",  # 新增scikit-learn
        "pyod"
    ]
    
    print(f"将安装以下包: {', '.join(required_packages)}")
    
    # 询问用户确认
    response = input("\n是否继续安装? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("安装已取消")
        return
    
    print("\n开始安装...")
    
    # 升级pip
    print("正在升级pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True, check=True)
        print("✅ pip升级成功")
    except:
        print("⚠️ pip升级失败，继续安装...")
    
    # 安装每个包
    failed_packages = []
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    # 显示结果
    print("\n" + "="*50)
    if failed_packages:
        print(f"❌ 安装完成，但以下包安装失败: {', '.join(failed_packages)}")
        print("\n请手动运行以下命令:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("✅ 所有依赖包安装成功!")
        print("\n现在可以运行异常检测工具了:")
        print("  python pyod_anomaly_detection_tool.py")
    
    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()