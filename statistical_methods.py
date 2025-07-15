import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings

class DescriptiveStatistics:
  """描述性统计类"""
  
  def __init__(self):
      self.statistics_functions = {
          '计数': lambda x: x.count(),
          '均值': lambda x: x.mean(),
          '标准差': lambda x: x.std(),
          '方差': lambda x: x.var(),
          '最小值': lambda x: x.min(),
          '最大值': lambda x: x.max(),
          '中位数': lambda x: x.median(),
          '第一四分位数': lambda x: x.quantile(0.25),
          '第三四分位数': lambda x: x.quantile(0.75),
          '偏度': lambda x: stats.skew(x.dropna()),
          '峰度': lambda x: stats.kurtosis(x.dropna()),
          '变异系数': lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan
      }
  
  def calculate_numeric_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
      """计算数值变量的描述性统计"""
      results = {}
      
      for col in data.columns:
          if data[col].dtype in ['int64', 'float64']:
              col_stats = {}
              for stat_name, stat_func in self.statistics_functions.items():
                  try:
                      col_stats[stat_name] = stat_func(data[col])
                  except:
                      col_stats[stat_name] = np.nan
              
              results[col] = col_stats
      
      return pd.DataFrame(results).T.round(4)
  
  def calculate_categorical_statistics(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
      """计算分类变量的描述性统计"""
      results = {}
      
      categorical_cols = data.select_dtypes(include=['object', 'category']).columns
      
      for col in categorical_cols:
          value_counts = data[col].value_counts()
          proportions = data[col].value_counts(normalize=True)
          
          col_stats = pd.DataFrame({
              '频数': value_counts,
              '比例': proportions,
              '百分比': proportions * 100
          }).round(4)
          
          results[col] = col_stats
      
      return results
  
  def calculate_confidence_interval(self, data: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
      """计算置信区间"""
      n = len(data.dropna())
      mean = data.mean()
      std_err = data.std() / np.sqrt(n)
      
      # t分布的临界值
      alpha = 1 - confidence
      t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
      
      margin_error = t_critical * std_err
      
      return (mean - margin_error, mean + margin_error)

class InferentialStatistics:
  """推断统计类"""
  
  def __init__(self):
      self.alpha = 0.05
  
  def one_sample_ttest(self, data: pd.Series, mu: float) -> Dict[str, Any]:
      """单样本t检验"""
      clean_data = data.dropna()
      
      t_stat, p_value = stats.ttest_1samp(clean_data, mu)
      
      result = {
          'test_name': '单样本t检验',
          'statistic': t_stat,
          'p_value': p_value,
          'degrees_of_freedom': len(clean_data) - 1,
          'sample_mean': clean_data.mean(),
          'hypothesized_mean': mu,
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_ttest_result(p_value, self.alpha)
      }
      
      return result
  
  def two_sample_ttest(self, group1: pd.Series, group2: pd.Series, 
                      equal_var: bool = True) -> Dict[str, Any]:
      """独立样本t检验"""
      clean_group1 = group1.dropna()
      clean_group2 = group2.dropna()
      
      if equal_var:
          t_stat, p_value = stats.ttest_ind(clean_group1, clean_group2)
          test_name = '独立样本t检验（等方差）'
      else:
          t_stat, p_value = stats.ttest_ind(clean_group1, clean_group2, equal_var=False)
          test_name = '独立样本t检验（不等方差）'
      
      result = {
          'test_name': test_name,
          'statistic': t_stat,
          'p_value': p_value,
          'group1_mean': clean_group1.mean(),
          'group2_mean': clean_group2.mean(),
          'group1_std': clean_group1.std(),
          'group2_std': clean_group2.std(),
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_ttest_result(p_value, self.alpha)
      }
      
      return result
  
  def paired_ttest(self, before: pd.Series, after: pd.Series) -> Dict[str, Any]:
      """配对样本t检验"""
      # 确保数据长度相同
      min_len = min(len(before), len(after))
      before_clean = before[:min_len]
      after_clean = after[:min_len]
      
      # 移除任一组有缺失值的配对
      mask = ~(before_clean.isna() | after_clean.isna())
      before_clean = before_clean[mask]
      after_clean = after_clean[mask]
      
      t_stat, p_value = stats.ttest_rel(before_clean, after_clean)
      
      difference = after_clean - before_clean
      
      result = {
          'test_name': '配对样本t检验',
          'statistic': t_stat,
          'p_value': p_value,
          'degrees_of_freedom': len(before_clean) - 1,
          'mean_difference': difference.mean(),
          'std_difference': difference.std(),
          'before_mean': before_clean.mean(),
          'after_mean': after_clean.mean(),
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_ttest_result(p_value, self.alpha)
      }
      
      return result
  
    def chi_square_test(self, observed: pd.DataFrame) -> Dict[str, Any]:
      """卡方检验"""
      chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
      
      result = {
          'test_name': '卡方独立性检验',
          'statistic': chi2_stat,
          'p_value': p_value,
          'degrees_of_freedom': dof,
          'expected_frequencies': expected,
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_chi_square_result(p_value, self.alpha),
          'cramers_v': self._calculate_cramers_v(chi2_stat, observed.sum().sum(), min(observed.shape) - 1)
      }
      
      return result
  
  def anova_one_way(self, *groups) -> Dict[str, Any]:
      """单因素方差分析"""
      # 清理数据
      clean_groups = [group.dropna() for group in groups]
      
      f_stat, p_value = stats.f_oneway(*clean_groups)
      
      # 计算组间和组内平方和
      overall_mean = np.concatenate(clean_groups).mean()
      
      ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for group in clean_groups)
      ss_within = sum(sum((x - group.mean())**2) for group in clean_groups for x in group)
      ss_total = ss_between + ss_within
      
      df_between = len(clean_groups) - 1
      df_within = sum(len(group) for group in clean_groups) - len(clean_groups)
      df_total = df_between + df_within
      
      ms_between = ss_between / df_between
      ms_within = ss_within / df_within
      
      eta_squared = ss_between / ss_total
      
      result = {
          'test_name': '单因素方差分析',
          'f_statistic': f_stat,
          'p_value': p_value,
          'df_between': df_between,
          'df_within': df_within,
          'ss_between': ss_between,
          'ss_within': ss_within,
          'ms_between': ms_between,
          'ms_within': ms_within,
          'eta_squared': eta_squared,
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_anova_result(p_value, self.alpha)
      }
      
      return result
  
  def correlation_test(self, x: pd.Series, y: pd.Series, method: str = 'pearson') -> Dict[str, Any]:
      """相关性检验"""
      # 移除缺失值
      mask = ~(x.isna() | y.isna())
      x_clean = x[mask]
      y_clean = y[mask]
      
      if method == 'pearson':
          corr_coef, p_value = stats.pearsonr(x_clean, y_clean)
          test_name = '皮尔逊相关检验'
      elif method == 'spearman':
          corr_coef, p_value = stats.spearmanr(x_clean, y_clean)
          test_name = '斯皮尔曼等级相关检验'
      elif method == 'kendall':
          corr_coef, p_value = stats.kendalltau(x_clean, y_clean)
          test_name = '肯德尔τ相关检验'
      else:
          raise ValueError("方法必须是 'pearson', 'spearman', 或 'kendall'")
      
      result = {
          'test_name': test_name,
          'correlation_coefficient': corr_coef,
          'p_value': p_value,
          'sample_size': len(x_clean),
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_correlation_result(corr_coef, p_value, self.alpha)
      }
      
      return result
  
  def normality_test(self, data: pd.Series) -> Dict[str, Any]:
      """正态性检验"""
      clean_data = data.dropna()
      
      results = {}
      
      # Shapiro-Wilk检验（适用于小样本）
      if len(clean_data) <= 5000:
          shapiro_stat, shapiro_p = stats.shapiro(clean_data)
          results['shapiro_wilk'] = {
              'statistic': shapiro_stat,
              'p_value': shapiro_p,
              'significant': shapiro_p < self.alpha
          }
      
      # Kolmogorov-Smirnov检验
      ks_stat, ks_p = stats.kstest(clean_data, 'norm', args=(clean_data.mean(), clean_data.std()))
      results['kolmogorov_smirnov'] = {
          'statistic': ks_stat,
          'p_value': ks_p,
          'significant': ks_p < self.alpha
      }
      
      # Anderson-Darling检验
      ad_result = stats.anderson(clean_data, dist='norm')
      results['anderson_darling'] = {
          'statistic': ad_result.statistic,
          'critical_values': ad_result.critical_values,
          'significance_levels': ad_result.significance_level
      }
      
      return results
  
  def _interpret_ttest_result(self, p_value: float, alpha: float) -> str:
      """解释t检验结果"""
      if p_value < alpha:
          return f"在α={alpha}水平下，拒绝原假设，差异具有统计学意义"
      else:
          return f"在α={alpha}水平下，不能拒绝原假设，差异无统计学意义"
  
  def _interpret_chi_square_result(self, p_value: float, alpha: float) -> str:
      """解释卡方检验结果"""
      if p_value < alpha:
          return f"在α={alpha}水平下，拒绝原假设，变量间存在关联性"
      else:
          return f"在α={alpha}水平下，不能拒绝原假设，变量间无显著关联"
  
  def _interpret_anova_result(self, p_value: float, alpha: float) -> str:
      """解释方差分析结果"""
      if p_value < alpha:
          return f"在α={alpha}水平下，拒绝原假设，组间差异显著"
      else:
          return f"在α={alpha}水平下，不能拒绝原假设，组间差异不显著"
  
  def _interpret_correlation_result(self, corr_coef: float, p_value: float, alpha: float) -> str:
      """解释相关性检验结果"""
      strength = self._get_correlation_strength(abs(corr_coef))
      direction = "正" if corr_coef > 0 else "负"
      
      if p_value < alpha:
          return f"存在{strength}的{direction}相关关系，具有统计学意义"
      else:
          return f"相关关系不具有统计学意义"
  
  def _get_correlation_strength(self, abs_corr: float) -> str:
      """获取相关强度描述"""
      if abs_corr >= 0.8:
          return "很强"
      elif abs_corr >= 0.6:
          return "强"
      elif abs_corr >= 0.4:
          return "中等"
      elif abs_corr >= 0.2:
          return "弱"
      else:
          return "很弱"
  
  def _calculate_cramers_v(self, chi2: float, n: int, min_dim: int) -> float:
      """计算Cramer's V系数"""
      return np.sqrt(chi2 / (n * min_dim))

class NonParametricTests:
  """非参数检验类"""
  
  def __init__(self):
      self.alpha = 0.05
  
  def mann_whitney_u_test(self, group1: pd.Series, group2: pd.Series) -> Dict[str, Any]:
      """Mann-Whitney U检验"""
      clean_group1 = group1.dropna()
      clean_group2 = group2.dropna()
      
      u_stat, p_value = stats.mannwhitneyu(clean_group1, clean_group2, alternative='two-sided')
      
      result = {
          'test_name': 'Mann-Whitney U检验',
          'u_statistic': u_stat,
          'p_value': p_value,
          'group1_median': clean_group1.median(),
          'group2_median': clean_group2.median(),
          'group1_size': len(clean_group1),
          'group2_size': len(clean_group2),
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_nonparametric_result(p_value, self.alpha, "两组")
      }
      
      return result
  
  def wilcoxon_signed_rank_test(self, before: pd.Series, after: pd.Series) -> Dict[str, Any]:
      """Wilcoxon符号秩检验"""
      # 确保数据长度相同并移除缺失值
      min_len = min(len(before), len(after))
      before_clean = before[:min_len]
      after_clean = after[:min_len]
      
      mask = ~(before_clean.isna() | after_clean.isna())
      before_clean = before_clean[mask]
      after_clean = after_clean[mask]
      
      w_stat, p_value = stats.wilcoxon(before_clean, after_clean)
      
      result = {
          'test_name': 'Wilcoxon符号秩检验',
          'w_statistic': w_stat,
          'p_value': p_value,
          'before_median': before_clean.median(),
          'after_median': after_clean.median(),
          'sample_size': len(before_clean),
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_nonparametric_result(p_value, self.alpha, "配对")
      }
      
      return result
  
  def kruskal_wallis_test(self, *groups) -> Dict[str, Any]:
      """Kruskal-Wallis检验"""
      clean_groups = [group.dropna() for group in groups]
      
      h_stat, p_value = stats.kruskal(*clean_groups)
      
      result = {
          'test_name': 'Kruskal-Wallis检验',
          'h_statistic': h_stat,
          'p_value': p_value,
          'degrees_of_freedom': len(clean_groups) - 1,
          'group_medians': [group.median() for group in clean_groups],
          'group_sizes': [len(group) for group in clean_groups],
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_nonparametric_result(p_value, self.alpha, "多组")
      }
      
      return result
  
  def friedman_test(self, *groups) -> Dict[str, Any]:
      """Friedman检验"""
      clean_groups = [group.dropna() for group in groups]
      
      # 确保所有组的长度相同
      min_len = min(len(group) for group in clean_groups)
      aligned_groups = [group[:min_len] for group in clean_groups]
      
      chi2_stat, p_value = stats.friedmanchisquare(*aligned_groups)
      
      result = {
          'test_name': 'Friedman检验',
          'chi2_statistic': chi2_stat,
          'p_value': p_value,
          'degrees_of_freedom': len(aligned_groups) - 1,
          'sample_size': min_len,
          'significant': p_value < self.alpha,
          'interpretation': self._interpret_nonparametric_result(p_value, self.alpha, "重复测量")
      }
      
      return result
  
  def _interpret_nonparametric_result(self, p_value: float, alpha: float, test_type: str) -> str:
      """解释非参数检验结果"""
      if p_value < alpha:
          return f"在α={alpha}水平下，拒绝原假设，{test_type}差异显著"
      else:
          return f"在α={alpha}水平下，不能拒绝原假设，{test_type}差异不显著"

class EffectSizeCalculator:
  """效应量计算器"""
  
  @staticmethod
  def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
      """计算Cohen's d"""
      n1, n2 = len(group1), len(group2)
      s1, s2 = group1.std(), group2.std()
      
      # 合并标准差
      pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
      
      return (group1.mean() - group2.mean()) / pooled_std
  
  @staticmethod
  def eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
      """计算eta平方"""
      return (f_stat * df_between) / (f_stat * df_between + df_within)
  
  @staticmethod
  def omega_squared(f_stat: float, df_between: int, df_within: int) -> float:
      """计算omega平方"""
      return (f_stat - 1) * df_between / (f_stat * df_between + df_within + 1)
  
  @staticmethod
  def r_squared_from_correlation(r: float) -> float:
      """从相关系数计算决定系数"""
      return r**2
  
  @staticmethod
  def interpret_cohens_d(d: float) -> str:
      """解释Cohen's d效应量"""
      abs_d = abs(d)
      if abs_d < 0.2:
          return "无效应或极小效应"
      elif abs_d < 0.5:
          return "小效应"
      elif abs_d < 0.8:
          return "中等效应"
      else:
          return "大效应"
