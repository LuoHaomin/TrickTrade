#!/usr/bin/env python3
"""
基金强化学习策略Demo运行脚本
简化版运行脚本，用于快速启动Demo
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'akshare', 'stable_baselines3', 'backtrader', 'talib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def run_demo():
    """运行Demo"""
    print("🚀 启动基金强化学习策略Demo...")
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 创建必要目录
    os.makedirs('demo/output', exist_ok=True)
    os.makedirs('demo/data', exist_ok=True)
    
    # 运行主程序
    try:
        result = subprocess.run([sys.executable, 'demo/main.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Demo运行成功！")
            print("\n📊 查看结果:")
            print("  - 回测结果: demo/output/results/")
            print("  - 可视化图表: demo/output/plots/")
            print("  - 训练日志: demo/output/logs/")
            return True
        else:
            print("❌ Demo运行失败:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行Demo时出错: {e}")
        return False

def run_test():
    """运行测试"""
    print("🧪 运行模块测试...")
    
    try:
        result = subprocess.run([sys.executable, 'demo/test_demo.py'], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ 所有测试通过！")
            return True
        else:
            print("❌ 部分测试失败")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基金强化学习策略Demo运行脚本')
    parser.add_argument('--test', action='store_true', help='仅运行测试')
    parser.add_argument('--check', action='store_true', help='仅检查依赖')
    
    args = parser.parse_args()
    
    if args.check:
        check_dependencies()
    elif args.test:
        run_test()
    else:
        # 默认运行完整Demo
        if run_test():
            print("\n" + "="*50)
            run_demo()
        else:
            print("❌ 测试失败，请先解决依赖问题")

if __name__ == "__main__":
    main()
