"""
皮尔森相关性分析器
分析基金间的线性相关性
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class PearsonCorrelationAnalyzer:
    """皮尔森相关性分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化皮尔森相关性分析器
        
        Args:
            config: 配置参数
                - min_periods: 最小观测期数
                - significance_level: 显著性水平
                - correlation_threshold: 相关性阈值
        """
        self.config = config or {}
        self.min_periods = self.config.get('min_periods', 30)
        self.significance_level = self.config.get('significance_level', 0.05)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.5)
        
    def calculate_correlation_matrix(self, data: pd.DataFrame, 
                                   price_column: str = 'Close') -> pd.DataFrame:
        """
        计算相关性矩阵
        
        Args:
            data: 多基金数据，列为基金代码，行为日期
            price_column: 价格列名
            
        Returns:
            相关性矩阵
        """
        # 确保数据是数值类型
        numeric_data = data.select_dtypes(include=[np.number])
        
        # 计算皮尔森相关系数
        correlation_matrix = numeric_data.corr(method='pearson', min_periods=self.min_periods)
        
        return correlation_matrix
    
    def calculate_pairwise_correlation(self, fund1_data: pd.Series, fund2_data: pd.Series,
                                     price_column: str = 'Close') -> Dict[str, Any]:
        """
        计算两个基金间的相关性
        
        Args:
            fund1_data: 基金1数据
            fund2_data: 基金2数据
            price_column: 价格列名
            
        Returns:
            相关性分析结果
        """
        # 提取价格数据
        if isinstance(fund1_data, pd.DataFrame):
            prices1 = fund1_data[price_column]
        else:
            prices1 = fund1_data
            
        if isinstance(fund2_data, pd.DataFrame):
            prices2 = fund2_data[price_column]
        else:
            prices2 = fund2_data
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'fund1': prices1,
            'fund2': prices2
        }).dropna()
        
        if len(aligned_data) < self.min_periods:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'is_significant': False,
                'sample_size': len(aligned_data),
                'error': 'Insufficient data'
            }
        
        # 计算相关系数和p值
        correlation, p_value = pearsonr(aligned_data['fund1'], aligned_data['fund2'])
        
        # 判断显著性
        is_significant = p_value < self.significance_level
        
        # 判断相关性强度
        strength = self._classify_correlation_strength(correlation)
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'is_significant': is_significant,
            'strength': strength,
            'sample_size': len(aligned_data),
            'interpretation': self._interpret_correlation(correlation, strength)
        }
    
    def find_highly_correlated_pairs(self, correlation_matrix: pd.DataFrame, 
                                   threshold: float = None) -> List[Dict[str, Any]]:
        """
        找出高相关性基金对
        
        Args:
            correlation_matrix: 相关性矩阵
            threshold: 相关性阈值
            
        Returns:
            高相关性基金对列表
        """
        if threshold is None:
            threshold = self.correlation_threshold
        
        # 获取上三角矩阵（避免重复）
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # 找出高相关性对
        high_corr_pairs = []
        
        for i in range(len(upper_triangle.columns)):
            for j in range(i+1, len(upper_triangle.columns)):
                corr_value = upper_triangle.iloc[i, j]
                
                if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        'fund1': upper_triangle.columns[i],
                        'fund2': upper_triangle.columns[j],
                        'correlation': corr_value,
                        'strength': self._classify_correlation_strength(corr_value),
                        'interpretation': self._interpret_correlation(corr_value, self._classify_correlation_strength(corr_value))
                    })
        
        # 按相关性绝对值排序
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return high_corr_pairs
    
    def find_negative_correlated_pairs(self, correlation_matrix: pd.DataFrame, 
                                     threshold: float = -0.5) -> List[Dict[str, Any]]:
        """
        找出负相关性基金对（对冲机会）
        
        Args:
            correlation_matrix: 相关性矩阵
            threshold: 负相关性阈值
            
        Returns:
            负相关性基金对列表
        """
        # 获取上三角矩阵
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # 找出负相关性对
        negative_corr_pairs = []
        
        for i in range(len(upper_triangle.columns)):
            for j in range(i+1, len(upper_triangle.columns)):
                corr_value = upper_triangle.iloc[i, j]
                
                if not pd.isna(corr_value) and corr_value <= threshold:
                    negative_corr_pairs.append({
                        'fund1': upper_triangle.columns[i],
                        'fund2': upper_triangle.columns[j],
                        'correlation': corr_value,
                        'strength': self._classify_correlation_strength(corr_value),
                        'interpretation': self._interpret_correlation(corr_value, self._classify_correlation_strength(corr_value)),
                        'hedge_potential': self._assess_hedge_potential(corr_value)
                    })
        
        # 按相关性排序（最负的在前）
        negative_corr_pairs.sort(key=lambda x: x['correlation'])
        
        return negative_corr_pairs
    
    def analyze_correlation_clusters(self, correlation_matrix: pd.DataFrame,
                                  threshold: float = 0.7) -> Dict[str, Any]:
        """
        分析相关性聚类
        
        Args:
            correlation_matrix: 相关性矩阵
            threshold: 聚类阈值
            
        Returns:
            聚类分析结果
        """
        from sklearn.cluster import AgglomerativeClustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        # 计算距离矩阵（1 - 相关性）
        distance_matrix = 1 - correlation_matrix.abs()
        
        # 使用层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,
            linkage='average',
            metric='precomputed'
        )
        
        # 移除缺失值
        valid_indices = ~distance_matrix.isna().any(axis=1)
        valid_distance_matrix = distance_matrix.loc[valid_indices, valid_indices]
        
        if len(valid_distance_matrix) < 2:
            return {'error': 'Insufficient data for clustering'}
        
        # 执行聚类
        cluster_labels = clustering.fit_predict(valid_distance_matrix)
        
        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            fund_name = valid_distance_matrix.index[i]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(fund_name)
        
        # 计算聚类内部平均相关性
        cluster_stats = {}
        for cluster_id, funds in clusters.items():
            if len(funds) > 1:
                cluster_corr_matrix = correlation_matrix.loc[funds, funds]
                avg_correlation = cluster_corr_matrix.where(
                    np.triu(np.ones(cluster_corr_matrix.shape), k=1).astype(bool)
                ).mean().mean()
                
                cluster_stats[cluster_id] = {
                    'funds': funds,
                    'size': len(funds),
                    'avg_correlation': avg_correlation,
                    'cohesion': self._assess_cluster_cohesion(avg_correlation)
                }
            else:
                cluster_stats[cluster_id] = {
                    'funds': funds,
                    'size': len(funds),
                    'avg_correlation': np.nan,
                    'cohesion': 'single_fund'
                }
        
        return {
            'clusters': cluster_stats,
            'n_clusters': len(clusters),
            'threshold': threshold,
            'clustering_method': 'hierarchical'
        }
    
    def calculate_rolling_correlation(self, fund1_data: pd.Series, fund2_data: pd.Series,
                                    window: int = 30, price_column: str = 'Close') -> pd.Series:
        """
        计算滚动相关性
        
        Args:
            fund1_data: 基金1数据
            fund2_data: 基金2数据
            window: 滚动窗口大小
            price_column: 价格列名
            
        Returns:
            滚动相关性序列
        """
        # 提取价格数据
        if isinstance(fund1_data, pd.DataFrame):
            prices1 = fund1_data[price_column]
        else:
            prices1 = fund1_data
            
        if isinstance(fund2_data, pd.DataFrame):
            prices2 = fund2_data[price_column]
        else:
            prices2 = fund2_data
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'fund1': prices1,
            'fund2': prices2
        }).dropna()
        
        # 计算滚动相关性
        rolling_corr = aligned_data['fund1'].rolling(window=window).corr(aligned_data['fund2'])
        
        return rolling_corr
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """分类相关性强度"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _interpret_correlation(self, correlation: float, strength: str) -> str:
        """解释相关性"""
        if pd.isna(correlation):
            return '无法计算相关性'
        
        direction = '正相关' if correlation > 0 else '负相关'
        
        interpretations = {
            'very_strong': f'极强{direction}',
            'strong': f'强{direction}',
            'moderate': f'中等{direction}',
            'weak': f'弱{direction}',
            'very_weak': f'极弱{direction}'
        }
        
        return interpretations.get(strength, '未知强度')
    
    def _assess_hedge_potential(self, correlation: float) -> str:
        """评估对冲潜力"""
        if pd.isna(correlation):
            return '无法评估'
        
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.8:
            return '高对冲潜力'
        elif abs_corr >= 0.6:
            return '中等对冲潜力'
        elif abs_corr >= 0.4:
            return '低对冲潜力'
        else:
            return '无对冲潜力'
    
    def _assess_cluster_cohesion(self, avg_correlation: float) -> str:
        """评估聚类内聚性"""
        if pd.isna(avg_correlation):
            return '无法评估'
        
        abs_corr = abs(avg_correlation)
        
        if abs_corr >= 0.8:
            return '高内聚性'
        elif abs_corr >= 0.6:
            return '中等内聚性'
        elif abs_corr >= 0.4:
            return '低内聚性'
        else:
            return '无内聚性'
    
    def generate_correlation_report(self, correlation_matrix: pd.DataFrame) -> str:
        """
        生成相关性分析报告
        
        Args:
            correlation_matrix: 相关性矩阵
            
        Returns:
            分析报告字符串
        """
        report = "=== 皮尔森相关性分析报告 ===\n\n"
        
        # 基本统计
        report += f"分析基金数量: {len(correlation_matrix)}\n"
        report += f"相关性阈值: {self.correlation_threshold}\n"
        report += f"显著性水平: {self.significance_level}\n\n"
        
        # 高相关性对
        high_corr_pairs = self.find_highly_correlated_pairs(correlation_matrix)
        report += f"高相关性基金对数量: {len(high_corr_pairs)}\n"
        
        if high_corr_pairs:
            report += "前5个高相关性对:\n"
            for i, pair in enumerate(high_corr_pairs[:5]):
                report += f"  {i+1}. {pair['fund1']} - {pair['fund2']}: {pair['correlation']:.3f} ({pair['strength']})\n"
        
        report += "\n"
        
        # 负相关性对
        negative_corr_pairs = self.find_negative_correlated_pairs(correlation_matrix)
        report += f"负相关性基金对数量: {len(negative_corr_pairs)}\n"
        
        if negative_corr_pairs:
            report += "前5个负相关性对:\n"
            for i, pair in enumerate(negative_corr_pairs[:5]):
                report += f"  {i+1}. {pair['fund1']} - {pair['fund2']}: {pair['correlation']:.3f} ({pair['hedge_potential']})\n"
        
        report += "\n"
        
        # 聚类分析
        cluster_result = self.analyze_correlation_clusters(correlation_matrix)
        if 'error' not in cluster_result:
            report += f"聚类数量: {cluster_result['n_clusters']}\n"
            report += "聚类详情:\n"
            for cluster_id, stats in cluster_result['clusters'].items():
                report += f"  聚类 {cluster_id}: {stats['size']} 个基金, 平均相关性: {stats['avg_correlation']:.3f}\n"
        
        return report
