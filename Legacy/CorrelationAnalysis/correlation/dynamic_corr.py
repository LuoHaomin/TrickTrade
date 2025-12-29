"""
动态相关性分析器
分析基金间随时间变化的相关性
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class DynamicCorrelationAnalyzer:
    """动态相关性分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化动态相关性分析器
        
        Args:
            config: 配置参数
                - window_size: 滚动窗口大小
                - step_size: 步长
                - min_periods: 最小观测期数
                - correlation_threshold: 相关性阈值
        """
        self.config = config or {}
        self.window_size = self.config.get('window_size', 30)
        self.step_size = self.config.get('step_size', 1)
        self.min_periods = self.config.get('min_periods', 20)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.5)
        
    def calculate_rolling_correlation(self, fund1_data: pd.Series, fund2_data: pd.Series,
                                    price_column: str = 'Close') -> pd.DataFrame:
        """
        计算滚动相关性
        
        Args:
            fund1_data: 基金1数据
            fund2_data: 基金2数据
            price_column: 价格列名
            
        Returns:
            滚动相关性结果
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
            return pd.DataFrame({'error': ['Insufficient data']})
        
        # 计算滚动相关性
        rolling_corr = aligned_data['fund1'].rolling(
            window=self.window_size, min_periods=self.min_periods
        ).corr(aligned_data['fund2'])
        
        # 计算滚动p值
        rolling_p_values = self._calculate_rolling_p_values(aligned_data)
        
        # 计算滚动置信区间
        rolling_ci = self._calculate_rolling_confidence_interval(rolling_corr, len(aligned_data))
        
        result = pd.DataFrame({
            'date': aligned_data.index,
            'correlation': rolling_corr,
            'p_value': rolling_p_values,
            'lower_ci': rolling_ci['lower'],
            'upper_ci': rolling_ci['upper'],
            'is_significant': rolling_p_values < 0.05
        })
        
        return result
    
    def calculate_time_varying_correlation(self, fund1_data: pd.Series, fund2_data: pd.Series,
                                         price_column: str = 'Close') -> Dict[str, Any]:
        """
        计算时变相关性
        
        Args:
            fund1_data: 基金1数据
            fund2_data: 基金2数据
            price_column: 价格列名
            
        Returns:
            时变相关性分析结果
        """
        # 计算滚动相关性
        rolling_result = self.calculate_rolling_correlation(fund1_data, fund2_data, price_column)
        
        if 'error' in rolling_result.columns:
            return {'error': 'Insufficient data'}
        
        # 分析相关性变化
        correlation_changes = self._analyze_correlation_changes(rolling_result)
        
        # 识别相关性突变点
        change_points = self._identify_change_points(rolling_result)
        
        # 分析相关性稳定性
        stability_metrics = self._analyze_correlation_stability(rolling_result)
        
        # 分析相关性趋势
        trend_analysis = self._analyze_correlation_trend(rolling_result)
        
        return {
            'rolling_correlation': rolling_result,
            'correlation_changes': correlation_changes,
            'change_points': change_points,
            'stability_metrics': stability_metrics,
            'trend_analysis': trend_analysis,
            'summary': self._generate_summary(rolling_result, correlation_changes, stability_metrics)
        }
    
    def analyze_correlation_regimes(self, fund1_data: pd.Series, fund2_data: pd.Series,
                                  price_column: str = 'Close') -> Dict[str, Any]:
        """
        分析相关性制度
        
        Args:
            fund1_data: 基金1数据
            fund2_data: 基金2数据
            price_column: 价格列名
            
        Returns:
            相关性制度分析结果
        """
        # 计算滚动相关性
        rolling_result = self.calculate_rolling_correlation(fund1_data, fund2_data, price_column)
        
        if 'error' in rolling_result.columns:
            return {'error': 'Insufficient data'}
        
        # 识别相关性制度
        regimes = self._identify_correlation_regimes(rolling_result)
        
        # 分析制度特征
        regime_characteristics = self._analyze_regime_characteristics(rolling_result, regimes)
        
        # 分析制度转换
        regime_transitions = self._analyze_regime_transitions(regimes)
        
        return {
            'regimes': regimes,
            'regime_characteristics': regime_characteristics,
            'regime_transitions': regime_transitions,
            'regime_summary': self._generate_regime_summary(regimes, regime_characteristics)
        }
    
    def calculate_conditional_correlation(self, fund1_data: pd.Series, fund2_data: pd.Series,
                                        condition_data: pd.Series, price_column: str = 'Close') -> Dict[str, Any]:
        """
        计算条件相关性
        
        Args:
            fund1_data: 基金1数据
            fund2_data: 基金2数据
            condition_data: 条件数据（如市场指数）
            price_column: 价格列名
            
        Returns:
            条件相关性分析结果
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
        
        if isinstance(condition_data, pd.DataFrame):
            condition_prices = condition_data[price_column]
        else:
            condition_prices = condition_data
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'fund1': prices1,
            'fund2': prices2,
            'condition': condition_prices
        }).dropna()
        
        if len(aligned_data) < self.min_periods:
            return {'error': 'Insufficient data'}
        
        # 计算条件相关性
        conditional_corr = self._calculate_conditional_correlation(aligned_data)
        
        # 分析条件相关性变化
        conditional_changes = self._analyze_conditional_changes(conditional_corr)
        
        return {
            'conditional_correlation': conditional_corr,
            'conditional_changes': conditional_changes,
            'summary': self._generate_conditional_summary(conditional_corr)
        }
    
    def _calculate_rolling_p_values(self, data: pd.DataFrame) -> pd.Series:
        """计算滚动p值"""
        p_values = []
        
        for i in range(len(data)):
            if i < self.window_size - 1:
                p_values.append(np.nan)
            else:
                start_idx = max(0, i - self.window_size + 1)
                end_idx = i + 1
                
                window_data = data.iloc[start_idx:end_idx]
                
                if len(window_data) >= self.min_periods:
                    _, p_value = pearsonr(window_data['fund1'], window_data['fund2'])
                    p_values.append(p_value)
                else:
                    p_values.append(np.nan)
        
        return pd.Series(p_values, index=data.index)
    
    def _calculate_rolling_confidence_interval(self, correlation: pd.Series, n: int) -> Dict[str, pd.Series]:
        """计算滚动置信区间"""
        # 简化的置信区间计算
        # 实际应用中可以使用更精确的方法
        
        # Fisher变换
        z_scores = 0.5 * np.log((1 + correlation) / (1 - correlation))
        
        # 标准误差
        se = 1 / np.sqrt(n - 3)
        
        # 95%置信区间
        z_lower = z_scores - 1.96 * se
        z_upper = z_scores + 1.96 * se
        
        # 逆Fisher变换
        lower_ci = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        upper_ci = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return {
            'lower': lower_ci,
            'upper': upper_ci
        }
    
    def _analyze_correlation_changes(self, rolling_result: pd.DataFrame) -> Dict[str, Any]:
        """分析相关性变化"""
        correlation = rolling_result['correlation'].dropna()
        
        if len(correlation) < 2:
            return {'error': 'Insufficient data'}
        
        # 计算变化率
        changes = correlation.diff()
        
        # 计算变化统计
        change_stats = {
            'mean_change': changes.mean(),
            'std_change': changes.std(),
            'max_change': changes.max(),
            'min_change': changes.min(),
            'volatility': changes.std(),
            'trend': 'increasing' if changes.mean() > 0 else 'decreasing'
        }
        
        # 识别大幅变化
        threshold = changes.std() * 2
        large_changes = changes[abs(changes) > threshold]
        
        return {
            'change_stats': change_stats,
            'large_changes': large_changes,
            'change_frequency': len(large_changes) / len(changes)
        }
    
    def _identify_change_points(self, rolling_result: pd.DataFrame) -> List[Dict[str, Any]]:
        """识别相关性突变点"""
        correlation = rolling_result['correlation'].dropna()
        
        if len(correlation) < 10:
            return []
        
        # 使用简化的突变点检测方法
        # 实际应用中可以使用更复杂的方法如CUSUM、Bai-Perron等
        
        change_points = []
        
        # 计算移动平均
        ma_short = correlation.rolling(window=5).mean()
        ma_long = correlation.rolling(window=20).mean()
        
        # 识别交叉点
        crossovers = []
        for i in range(1, len(ma_short)):
            if (ma_short.iloc[i-1] <= ma_long.iloc[i-1] and ma_short.iloc[i] > ma_long.iloc[i]) or \
               (ma_short.iloc[i-1] >= ma_long.iloc[i-1] and ma_short.iloc[i] < ma_long.iloc[i]):
                crossovers.append(i)
        
        # 分析交叉点
        for idx in crossovers:
            if idx < len(correlation):
                change_points.append({
                    'date': correlation.index[idx],
                    'correlation': correlation.iloc[idx],
                    'type': 'crossover',
                    'significance': 'medium'
                })
        
        return change_points
    
    def _analyze_correlation_stability(self, rolling_result: pd.DataFrame) -> Dict[str, float]:
        """分析相关性稳定性"""
        correlation = rolling_result['correlation'].dropna()
        
        if len(correlation) < 2:
            return {'error': 'Insufficient data'}
        
        # 计算稳定性指标
        stability_metrics = {
            'variance': correlation.var(),
            'coefficient_of_variation': correlation.std() / abs(correlation.mean()) if correlation.mean() != 0 else np.inf,
            'range': correlation.max() - correlation.min(),
            'stability_index': 1 / (1 + correlation.std()),  # 自定义稳定性指标
            'persistence': self._calculate_persistence(correlation)
        }
        
        return stability_metrics
    
    def _analyze_correlation_trend(self, rolling_result: pd.DataFrame) -> Dict[str, Any]:
        """分析相关性趋势"""
        correlation = rolling_result['correlation'].dropna()
        
        if len(correlation) < 5:
            return {'error': 'Insufficient data'}
        
        # 线性趋势
        x = np.arange(len(correlation))
        slope, intercept = np.polyfit(x, correlation, 1)
        
        # 趋势强度
        trend_strength = abs(slope) * len(correlation)
        
        # 趋势方向
        trend_direction = 'increasing' if slope > 0 else 'decreasing'
        
        return {
            'slope': slope,
            'intercept': intercept,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'trend_significance': 'strong' if trend_strength > 0.1 else 'weak'
        }
    
    def _identify_correlation_regimes(self, rolling_result: pd.DataFrame) -> List[Dict[str, Any]]:
        """识别相关性制度"""
        correlation = rolling_result['correlation'].dropna()
        
        if len(correlation) < 20:
            return []
        
        # 简化的制度识别方法
        # 实际应用中可以使用更复杂的方法如Markov Switching等
        
        regimes = []
        
        # 定义制度阈值
        high_threshold = correlation.quantile(0.75)
        low_threshold = correlation.quantile(0.25)
        
        current_regime = None
        regime_start = None
        
        for i, (date, corr) in enumerate(correlation.items()):
            if corr > high_threshold:
                regime_type = 'high'
            elif corr < low_threshold:
                regime_type = 'low'
            else:
                regime_type = 'medium'
            
            if current_regime != regime_type:
                # 制度转换
                if current_regime is not None:
                    regimes.append({
                        'regime': current_regime,
                        'start_date': regime_start,
                        'end_date': date,
                        'duration': i - regime_start,
                        'avg_correlation': correlation.iloc[regime_start:i].mean()
                    })
                
                current_regime = regime_type
                regime_start = i
        
        # 添加最后一个制度
        if current_regime is not None:
            regimes.append({
                'regime': current_regime,
                'start_date': regime_start,
                'end_date': len(correlation) - 1,
                'duration': len(correlation) - regime_start,
                'avg_correlation': correlation.iloc[regime_start:].mean()
            })
        
        return regimes
    
    def _analyze_regime_characteristics(self, rolling_result: pd.DataFrame, regimes: List[Dict]) -> Dict[str, Any]:
        """分析制度特征"""
        if not regimes:
            return {'error': 'No regimes identified'}
        
        regime_stats = {}
        
        for regime_info in regimes:
            regime_type = regime_info['regime']
            
            if regime_type not in regime_stats:
                regime_stats[regime_type] = {
                    'count': 0,
                    'total_duration': 0,
                    'avg_correlation': [],
                    'avg_duration': 0
                }
            
            regime_stats[regime_type]['count'] += 1
            regime_stats[regime_type]['total_duration'] += regime_info['duration']
            regime_stats[regime_type]['avg_correlation'].append(regime_info['avg_correlation'])
        
        # 计算平均值
        for regime_type in regime_stats:
            stats = regime_stats[regime_type]
            stats['avg_duration'] = stats['total_duration'] / stats['count']
            stats['avg_correlation'] = np.mean(stats['avg_correlation'])
        
        return regime_stats
    
    def _analyze_regime_transitions(self, regimes: List[Dict]) -> List[Dict[str, Any]]:
        """分析制度转换"""
        transitions = []
        
        for i in range(1, len(regimes)):
            prev_regime = regimes[i-1]
            curr_regime = regimes[i]
            
            transition = {
                'from_regime': prev_regime['regime'],
                'to_regime': curr_regime['regime'],
                'transition_date': curr_regime['start_date'],
                'correlation_change': curr_regime['avg_correlation'] - prev_regime['avg_correlation'],
                'transition_type': f"{prev_regime['regime']}_to_{curr_regime['regime']}"
            }
            
            transitions.append(transition)
        
        return transitions
    
    def _calculate_conditional_correlation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算条件相关性"""
        # 简化的条件相关性计算
        # 实际应用中可以使用更复杂的方法如DCC-GARCH等
        
        # 按条件变量分组
        condition_quantiles = pd.qcut(data['condition'], q=3, labels=['low', 'medium', 'high'])
        
        conditional_corr = {}
        
        for condition in ['low', 'medium', 'high']:
            condition_data = data[condition_quantiles == condition]
            
            if len(condition_data) >= self.min_periods:
                corr, p_value = pearsonr(condition_data['fund1'], condition_data['fund2'])
                conditional_corr[condition] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'sample_size': len(condition_data)
                }
            else:
                conditional_corr[condition] = {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'sample_size': len(condition_data)
                }
        
        return conditional_corr
    
    def _analyze_conditional_changes(self, conditional_corr: Dict[str, Any]) -> Dict[str, Any]:
        """分析条件相关性变化"""
        correlations = [info['correlation'] for info in conditional_corr.values() if not pd.isna(info['correlation'])]
        
        if len(correlations) < 2:
            return {'error': 'Insufficient data'}
        
        # 计算变化
        changes = {
            'low_to_medium': correlations[1] - correlations[0] if len(correlations) > 1 else np.nan,
            'medium_to_high': correlations[2] - correlations[1] if len(correlations) > 2 else np.nan,
            'low_to_high': correlations[2] - correlations[0] if len(correlations) > 2 else np.nan
        }
        
        return {
            'changes': changes,
            'max_change': max([abs(change) for change in changes.values() if not pd.isna(change)]),
            'change_pattern': self._classify_change_pattern(changes)
        }
    
    def _calculate_persistence(self, correlation: pd.Series) -> float:
        """计算相关性持续性"""
        if len(correlation) < 2:
            return 0.0
        
        # 计算自相关
        autocorr = correlation.autocorr(lag=1)
        
        return autocorr if not pd.isna(autocorr) else 0.0
    
    def _classify_change_pattern(self, changes: Dict[str, float]) -> str:
        """分类变化模式"""
        # 简化的模式分类
        if all(change > 0 for change in changes.values() if not pd.isna(change)):
            return 'monotonic_increasing'
        elif all(change < 0 for change in changes.values() if not pd.isna(change)):
            return 'monotonic_decreasing'
        else:
            return 'non_monotonic'
    
    def _generate_summary(self, rolling_result: pd.DataFrame, correlation_changes: Dict, 
                         stability_metrics: Dict) -> str:
        """生成摘要"""
        correlation = rolling_result['correlation'].dropna()
        
        summary = f"""
动态相关性分析摘要:
- 平均相关性: {correlation.mean():.3f}
- 相关性标准差: {correlation.std():.3f}
- 相关性范围: [{correlation.min():.3f}, {correlation.max():.3f}]
- 稳定性指数: {stability_metrics.get('stability_index', 0):.3f}
- 变化频率: {correlation_changes.get('change_frequency', 0):.3f}
        """
        
        return summary.strip()
    
    def _generate_regime_summary(self, regimes: List[Dict], regime_characteristics: Dict) -> str:
        """生成制度摘要"""
        if not regimes:
            return "未识别到相关性制度"
        
        summary = f"""
相关性制度分析摘要:
- 制度数量: {len(regimes)}
- 制度特征: {regime_characteristics}
- 平均制度持续时间: {np.mean([r['duration'] for r in regimes]):.1f}
        """
        
        return summary.strip()
    
    def _generate_conditional_summary(self, conditional_corr: Dict[str, Any]) -> str:
        """生成条件相关性摘要"""
        summary = "条件相关性分析摘要:\n"
        
        for condition, info in conditional_corr.items():
            if not pd.isna(info['correlation']):
                summary += f"- {condition}条件: 相关性={info['correlation']:.3f}, 样本数={info['sample_size']}\n"
        
        return summary.strip()
