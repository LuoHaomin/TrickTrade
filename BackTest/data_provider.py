"""
数据引入接口
提供统一的数据获取和预处理功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import akshare as ak
import yfinance as yf

class DataProvider:
    """数据提供者基类"""
    
    def __init__(self):
        self.data_cache = {}
    
    def get_data(self, symbol: str, start_date: str, end_date: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        获取数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        raise NotImplementedError
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        # 确保索引是日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            elif '日期' in data.columns:
                data['日期'] = pd.to_datetime(data['日期'])
                data = data.set_index('日期')
        
        # 确保数据按日期排序
        data = data.sort_index()
        
        # 移除缺失值
        data = data.dropna()
        
        return data

class AKShareDataProvider(DataProvider):
    """AKShare数据提供者"""
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 获取股票历史数据
            data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                    start_date=start_date.replace('-', ''), 
                                    end_date=end_date.replace('-', ''), 
                                    adjust="")
            
            if data.empty:
                return None
            
            # 数据预处理
            data = self._preprocess_data(data)
            
            # 重命名列以匹配标准格式
            data = data.rename(columns={
                '开盘': 'Open',
                '最高': 'High',
                '最低': 'Low',
                '收盘': 'Close',
                '成交量': 'Volume'
            })
            
            # 选择需要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return data[required_columns]
            
        except Exception as e:
            print(f"获取股票数据失败: {e}")
            return None
    
    def get_fund_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取基金数据
        
        Args:
            symbol: 基金代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 获取ETF基金历史数据
            data = ak.fund_etf_hist_em(symbol=symbol, period="daily", 
                                      start_date=start_date.replace('-', ''), 
                                      end_date=end_date.replace('-', ''), 
                                      adjust="")
            
            if data.empty:
                return None
            
            # 数据预处理
            data = self._preprocess_data(data)
            
            # 重命名列以匹配标准格式
            data = data.rename(columns={
                '开盘': 'Open',
                '最高': 'High',
                '最低': 'Low',
                '收盘': 'Close',
                '成交量': 'Volume'
            })
            
            # 选择需要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return data[required_columns]
            
        except Exception as e:
            print(f"获取基金数据失败: {e}")
            return None
    
    def get_index_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取指数数据
        
        Args:
            symbol: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 获取指数历史数据
            data = ak.stock_zh_index_daily(symbol=symbol)
            
            if data.empty:
                return None
            
            # 数据预处理
            data = self._preprocess_data(data)
            
            # 筛选日期范围
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # 重命名列以匹配标准格式
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # 选择需要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return data[required_columns]
            
        except Exception as e:
            print(f"获取指数数据失败: {e}")
            return None

class YahooFinanceDataProvider(DataProvider):
    """Yahoo Finance数据提供者"""
    
    def get_data(self, symbol: str, start_date: str, end_date: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        获取数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 获取数据
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if data.empty:
                return None
            
            # 数据预处理
            data = self._preprocess_data(data)
            
            # 重命名列以匹配标准格式
            data = data.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # 选择需要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return data[required_columns]
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

class MockDataProvider(DataProvider):
    """模拟数据提供者"""
    
    def __init__(self, seed: int = 42):
        super().__init__()
        np.random.seed(seed)
    
    def get_data(self, symbol: str, start_date: str, end_date: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        生成模拟数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        # 生成日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 生成模拟价格数据
        initial_price = kwargs.get('initial_price', 100.0)
        volatility = kwargs.get('volatility', 0.02)
        
        price = initial_price
        prices = []
        
        for _ in range(len(dates)):
            # 随机游走
            change = np.random.normal(0, volatility)
            price += change
            price = max(price, 0.01)  # 确保价格为正
            prices.append(price)
        
        # 生成OHLCV数据
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return data

class DataManager:
    """数据管理器"""
    
    def __init__(self):
        self.providers = {
            'akshare': AKShareDataProvider(),
            'yfinance': YahooFinanceDataProvider(),
            'mock': MockDataProvider()
        }
        self.default_provider = 'akshare'
    
    def get_data(self, symbol: str, start_date: str, end_date: str, 
                provider: str = None, data_type: str = 'auto', **kwargs) -> Optional[pd.DataFrame]:
        """
        获取数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            provider: 数据提供者
            data_type: 数据类型 ('stock', 'fund', 'index', 'auto')
            **kwargs: 其他参数
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        if provider is None:
            provider = self.default_provider
        
        if provider not in self.providers:
            raise ValueError(f"不支持的数据提供者: {provider}")
        
        data_provider = self.providers[provider]
        
        # 根据数据类型选择相应的方法
        if data_type == 'auto':
            # 自动判断数据类型
            if symbol.startswith(('sh', 'sz', '000', '002', '300', '600', '601', '603')):
                data_type = 'stock'
            elif symbol.startswith(('510', '159', '512')):
                data_type = 'fund'
            else:
                data_type = 'stock'  # 默认
        
        if data_type == 'stock':
            if provider == 'akshare':
                return data_provider.get_stock_data(symbol, start_date, end_date)
            else:
                return data_provider.get_data(symbol, start_date, end_date, **kwargs)
        elif data_type == 'fund':
            if provider == 'akshare':
                return data_provider.get_fund_data(symbol, start_date, end_date)
            else:
                return data_provider.get_data(symbol, start_date, end_date, **kwargs)
        elif data_type == 'index':
            if provider == 'akshare':
                return data_provider.get_index_data(symbol, start_date, end_date)
            else:
                return data_provider.get_data(symbol, start_date, end_date, **kwargs)
        else:
            return data_provider.get_data(symbol, start_date, end_date, **kwargs)
    
    def add_provider(self, name: str, provider: DataProvider):
        """添加数据提供者"""
        self.providers[name] = provider
    
    def set_default_provider(self, provider: str):
        """设置默认数据提供者"""
        if provider not in self.providers:
            raise ValueError(f"不支持的数据提供者: {provider}")
        self.default_provider = provider

# 全局数据管理器实例
data_manager = DataManager()

def get_data(symbol: str, start_date: str, end_date: str, 
            provider: str = None, data_type: str = 'auto', **kwargs) -> Optional[pd.DataFrame]:
    """
    获取数据的便捷函数
    
    Args:
        symbol: 标的代码
        start_date: 开始日期
        end_date: 结束日期
        provider: 数据提供者
        data_type: 数据类型
        **kwargs: 其他参数
        
    Returns:
        包含OHLCV数据的DataFrame
    """
    return data_manager.get_data(symbol, start_date, end_date, provider, data_type, **kwargs)
