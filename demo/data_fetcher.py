"""
ETF基金数据获取模块
使用AKShare API获取ETF历史数据，支持数据缓存和预处理
"""

import os
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETFDataFetcher:
    """ETF数据获取器"""
    
    def __init__(self, cache_dir: str = "demo/data"):
        """
        初始化数据获取器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 常用ETF代码映射
        self.etf_symbols = {
            "510300": "沪深300ETF",
            "510500": "中证500ETF", 
            "510050": "上证50ETF",
            "159919": "沪深300ETF",
            "159915": "创业板ETF",
            "512100": "中证1000ETF"
        }
    
    def get_etf_data(self, symbol: str, start_date: str, end_date: str, 
                     use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        获取ETF历史数据
        
        Args:
            symbol: ETF代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            use_cache: 是否使用缓存
            
        Returns:
            包含OHLCV数据的DataFrame，索引为日期
        """
        # 检查缓存
        if use_cache:
            cached_data = self._load_from_cache(symbol, start_date, end_date)
            if cached_data is not None:
                logger.info(f"从缓存加载 {symbol} 数据")
                return cached_data
        
        try:
            logger.info(f"从AKShare获取 {symbol} 数据: {start_date} 到 {end_date}")
            
            # 转换日期格式
            start_date_ak = start_date.replace("-", "")
            end_date_ak = end_date.replace("-", "")
            
            # 获取ETF历史数据
            if symbol.startswith("51") or symbol.startswith("52"):
                # 上交所ETF
                data = ak.fund_etf_hist_em(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date_ak,
                    end_date=end_date_ak,
                    adjust=""
                )
            else:
                # 深交所ETF
                data = ak.fund_etf_hist_em(
                    symbol=symbol,
                    period="daily", 
                    start_date=start_date_ak,
                    end_date=end_date_ak,
                    adjust=""
                )
            
            if data.empty:
                logger.warning(f"未获取到 {symbol} 的数据")
                return None
            
            # 数据预处理
            processed_data = self._preprocess_data(data)
            
            # 保存到缓存
            if use_cache:
                self._save_to_cache(symbol, start_date, end_date, processed_data)
            
            logger.info(f"成功获取 {symbol} 数据，共 {len(processed_data)} 条记录")
            return processed_data
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        # 重命名列
        column_mapping = {
            '开盘': 'Open',
            '最高': 'High', 
            '最低': 'Low',
            '收盘': 'Close',
            '成交量': 'Volume',
            '成交额': 'Amount',
            '振幅': 'Amplitude',
            '涨跌幅': 'Change',
            '涨跌额': 'ChangeAmount',
            '换手率': 'Turnover'
        }
        
        # 只保留需要的列
        available_columns = {}
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                available_columns[old_col] = new_col
        
        data = data.rename(columns=available_columns)
        
        # 设置日期索引
        if '日期' in data.columns:
            data['日期'] = pd.to_datetime(data['日期'])
            data = data.set_index('日期')
        elif data.index.name == '日期' or isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # 确保有必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"缺少必要列: {missing_columns}")
            # 尝试从现有列推断
            if 'Close' in data.columns and 'Volume' not in data.columns:
                data['Volume'] = 0  # 如果没有成交量数据，设为0
        
        # 数据清洗
        data = data.dropna(subset=['Close'])  # 删除收盘价缺失的行
        
        # 确保数值列是数值类型
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 删除异常值（价格为负或为0）
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # 按日期排序
        data = data.sort_index()
        
        # 填充缺失的成交量
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].fillna(0)
        
        return data
    
    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> str:
        """获取缓存文件路径"""
        filename = f"{symbol}_{start_date}_{end_date}.parquet"
        return os.path.join(self.cache_dir, filename)
    
    def _load_from_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        
        if os.path.exists(cache_path):
            try:
                data = pd.read_parquet(cache_path)
                # 检查缓存是否过期（24小时）
                cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - cache_time < timedelta(hours=24):
                    return data
                else:
                    logger.info(f"缓存已过期，将重新获取数据")
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        return None
    
    def _save_to_cache(self, symbol: str, start_date: str, end_date: str, data: pd.DataFrame):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        
        try:
            data.to_parquet(cache_path)
            logger.info(f"数据已缓存到: {cache_path}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def get_multiple_etf_data(self, symbols: list, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        获取多个ETF的数据
        
        Args:
            symbols: ETF代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            字典，键为ETF代码，值为DataFrame
        """
        data_dict = {}
        
        for symbol in symbols:
            logger.info(f"获取 {symbol} 数据...")
            data = self.get_etf_data(symbol, start_date, end_date)
            if data is not None:
                data_dict[symbol] = data
            else:
                logger.warning(f"跳过 {symbol}，数据获取失败")
        
        return data_dict
    
    def get_etf_list(self) -> pd.DataFrame:
        """
        获取ETF列表
        
        Returns:
            包含ETF基本信息的DataFrame
        """
        try:
            logger.info("获取ETF列表...")
            etf_list = ak.fund_etf_spot_em()
            logger.info(f"获取到 {len(etf_list)} 个ETF")
            return etf_list
        except Exception as e:
            logger.error(f"获取ETF列表失败: {e}")
            return pd.DataFrame()


def main():
    """测试函数"""
    fetcher = ETFDataFetcher()
    
    # 测试获取单个ETF数据
    symbol = "510300"  # 沪深300ETF
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    
    print(f"测试获取 {symbol} 数据...")
    data = fetcher.get_etf_data(symbol, start_date, end_date)
    
    if data is not None:
        print(f"数据形状: {data.shape}")
        print(f"数据列: {data.columns.tolist()}")
        print(f"日期范围: {data.index.min()} 到 {data.index.max()}")
        print("\n前5行数据:")
        print(data.head())
        print("\n数据统计:")
        print(data.describe())
    else:
        print("数据获取失败")


if __name__ == "__main__":
    main()
