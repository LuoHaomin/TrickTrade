import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

# fund_etf_spot_em_df = ak.fund_etf_spot_em()
# print(fund_etf_spot_em_df)

# 获取基金代码
# fund_code = fund_etf_spot_em_df['代码']
# print(fund_code)

# 获取代码为code的基金的历史数据
def get_fund_by_code(code, period="daily", start_date="20240101", end_date="20250101", adjust=""):
    try:
        fund_etf_hist_em_df = ak.fund_etf_hist_em(symbol=code, period=period, start_date=start_date, end_date=end_date, adjust=adjust)
        print(f"基金代码 {code} 的历史数据:")
        print(fund_etf_hist_em_df)
        return fund_etf_hist_em_df
    except Exception as e:
        print(f"获取基金代码 {code} 数据时出错: {e}")
        return None

# 数据可视化函数
def visualize_fund_data(fund_data, fund_code, fund_name=""):
    """
    可视化基金数据
    """
    if fund_data is None or fund_data.empty:
        print(f"基金代码 {fund_code} 没有数据可供可视化")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 确保日期列是datetime类型
    if '日期' in fund_data.columns:
        fund_data['日期'] = pd.to_datetime(fund_data['日期'])
        dates = fund_data['日期']
    else:
        dates = pd.date_range(start='2022-01-01', periods=len(fund_data), freq='D')
    
    # 绘制价格走势图
    if '收盘' in fund_data.columns:
        ax1.plot(dates, fund_data['收盘'], label='收盘价', linewidth=2, color='#1f77b4')
    if '开盘' in fund_data.columns:
        ax1.plot(dates, fund_data['开盘'], label='开盘价', alpha=0.7, color='#ff7f0e')
    if '最高' in fund_data.columns:
        ax1.plot(dates, fund_data['最高'], label='最高价', alpha=0.5, color='#2ca02c')
    if '最低' in fund_data.columns:
        ax1.plot(dates, fund_data['最低'], label='最低价', alpha=0.5, color='#d62728')
    
    ax1.set_title(f'{fund_name} ({fund_code}) 价格走势图', fontsize=16, fontweight='bold')
    ax1.set_ylabel('价格 (元)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 绘制成交量图
    if '成交量' in fund_data.columns:
        ax2.bar(dates, fund_data['成交量'], alpha=0.7, color='#9467bd', width=1)
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 打印数据统计信息
    print(f"\n{fund_name} ({fund_code}) 数据统计:")
    print(f"数据期间: {dates.min().strftime('%Y-%m-%d')} 至 {dates.max().strftime('%Y-%m-%d')}")
    print(f"数据点数: {len(fund_data)}")
    if '收盘' in fund_data.columns:
        print(f"最高价: {fund_data['收盘'].max():.2f} 元")
        print(f"最低价: {fund_data['收盘'].min():.2f} 元")
        print(f"平均价: {fund_data['收盘'].mean():.2f} 元")
        print(f"价格波动: {((fund_data['收盘'].max() - fund_data['收盘'].min()) / fund_data['收盘'].min() * 100):.2f}%")

# 测试不同的基金代码
print("\n测试基金代码 510050 (上证50ETF):")
result2 = get_fund_by_code(code="510050", period="daily", start_date="20220101", end_date="20230101", adjust="")
print(f"结果2: {result2 is not None}")

# 可视化数据
if result2 is not None:
    visualize_fund_data(result2, "510050", "上证50ETF")

# 多基金对比可视化函数
def compare_multiple_funds(fund_codes, fund_names, period="daily", start_date="20220101", end_date="20230101"):
    """
    对比多个基金的价格走势
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (code, name) in enumerate(zip(fund_codes, fund_names)):
        try:
            fund_data = ak.fund_etf_hist_em(symbol=code, period=period, start_date=start_date, end_date=end_date, adjust="")
            if not fund_data.empty and '收盘' in fund_data.columns:
                fund_data['日期'] = pd.to_datetime(fund_data['日期'])
                # 归一化价格（以第一个交易日为基准）
                normalized_price = fund_data['收盘'] / fund_data['收盘'].iloc[0] * 100
                ax.plot(fund_data['日期'], normalized_price, 
                       label=f'{name} ({code})', linewidth=2, color=colors[i % len(colors)])
                print(f"成功获取 {name} ({code}) 数据: {len(fund_data)} 个交易日")
            else:
                print(f"无法获取 {name} ({code}) 的数据")
        except Exception as e:
            print(f"获取 {name} ({code}) 数据时出错: {e}")
    
    ax.set_title('多基金价格走势对比 (归一化)', fontsize=16, fontweight='bold')
    ax.set_ylabel('归一化价格 (基准=100)', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

# 测试多基金对比
print("\n=== 多基金对比分析 ===")
fund_codes = ["510050", "159919", "512100", "515050"]  # 上证50ETF, 沪深300ETF, 中证1000ETF, 5GETF
fund_names = ["上证50ETF", "沪深300ETF", "中证1000ETF", "5GETF"]

compare_multiple_funds(fund_codes, fund_names)

