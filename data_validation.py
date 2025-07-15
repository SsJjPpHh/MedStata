import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings

class DataValidator:
  """数据验证器类"""
  
  def __init__(self):
      self.validation_rules = {
          'missing_data': self._check_missing_data,
          'data_types': self._check_data_types,
          'outliers': self._check_outliers,
          'duplicates': self._check_duplicates,
          'consistency': self._check_consistency
      }
  
  def validate_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
      """
      验证整个数据集
      
      Args:
          data: 待验证的数据框
          
      Returns:
          验证结果字典
      """
      results = {
          'is_valid': True,
          'issues': {},
          'warnings': [],
          'quality_score': 100
      }
      
      penalty_points = 0
      
      # 执行各项验证
      for rule_name, rule_func in self.validation_rules.items():
          try:
              rule_result = rule_func(data)
              results['issues'][rule_name] = rule_result['issues']
              results['warnings'].extend(rule_result['warnings'])
              penalty_points += rule_result['penalty']
              
              if rule_result['issues']:
                  results['is_valid'] = False
                  
          except Exception as e:
              results['warnings'].append(f"验证规则 {rule_name} 执行失败: {str(e)}")
      
      # 计算质量评分
      results['quality_score'] = max(0, 100 - penalty_points)
      
      return results
  
  def _check_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
      """检查缺失数据"""
      issues = []
      warnings = []
      penalty = 0
      
      # 计算缺失率
      missing_counts = data.isnull().sum()
      missing_rates = (missing_counts / len(data)) * 100
      
      for col, rate in missing_rates.items():
          if rate > 50:
              issues.append(f"列 '{col}' 缺失率过高: {rate:.1f}%")
              penalty += 20
          elif rate > 20:
              warnings.append(f"列 '{col}' 缺失率较高: {rate:.1f}%")
              penalty += 10
          elif rate > 5:
              warnings.append(f"列 '{col}' 存在缺失值: {rate:.1f}%")
              penalty += 5
      
      # 检查完全缺失的列
      completely_missing = missing_counts[missing_counts == len(data)]
      for col in completely_missing.index:
          issues.append(f"列 '{col}' 完全缺失")
          penalty += 30
      
      return {
          'issues': issues,
          'warnings': warnings,
          'penalty': penalty
      }
  
  def _check_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
      """检查数据类型"""
      issues = []
      warnings = []
      penalty = 0
      
      for col in data.columns:
          col_data = data[col].dropna()
          
          if len(col_data) == 0:
              continue
          
          # 检查数值列中的非数值数据
          if data[col].dtype in ['object', 'string']:
              # 尝试转换为数值
              numeric_convertible = pd.to_numeric(col_data, errors='coerce')
              if not numeric_convertible.isna().all():
                  non_numeric_count = numeric_convertible.isna().sum()
                  if non_numeric_count > 0:
                      warnings.append(
                          f"列 '{col}' 包含 {non_numeric_count} 个无法转换为数值的值"
                      )
                      penalty += 5
          
          # 检查日期格式
          if 'date' in col.lower() or 'time' in col.lower():
              try:
                  pd.to_datetime(col_data)
              except:
                  warnings.append(f"列 '{col}' 似乎是日期列但格式不标准")
                  penalty += 5
      
      return {
          'issues': issues,
          'warnings': warnings,
          'penalty': penalty
      }
  
  def _check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
      """检查异常值"""
      issues = []
      warnings = []
      penalty = 0
      
      numeric_cols = data.select_dtypes(include=[np.number]).columns
      
      for col in numeric_cols:
          col_data = data[col].dropna()
          
          if len(col_data) < 10:  # 数据太少无法检测异常值
              continue
          
          # 使用IQR方法检测异常值
          Q1 = col_data.quantile(0.25)
          Q3 = col_data.quantile(0.75)
          IQR = Q3 - Q1
          
          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR
          
          outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
          outlier_rate = len(outliers) / len(col_data) * 100
          
          if outlier_rate > 10:
              issues.append(f"列 '{col}' 异常值比例过高: {outlier_rate:.1f}%")
              penalty += 15
          elif outlier_rate > 5:
              warnings.append(f"列 '{col}' 存在较多异常值: {outlier_rate:.1f}%")
              penalty += 8
          elif outlier_rate > 0:
              warnings.append(f"列 '{col}' 存在异常值: {len(outliers)} 个")
              penalty += 3
      
      return {
          'issues': issues,
          'warnings': warnings,
          'penalty': penalty
      }
  
  def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
      """检查重复数据"""
      issues = []
      warnings = []
      penalty = 0
      
      # 检查完全重复的行
      duplicate_rows = data.duplicated().sum()
      if duplicate_rows > 0:
          duplicate_rate = duplicate_rows / len(data) * 100
          if duplicate_rate > 10:
              issues.append(f"重复行比例过高: {duplicate_rate:.1f}% ({duplicate_rows} 行)")
              penalty += 20
          elif duplicate_rate > 1:
              warnings.append(f"存在重复行: {duplicate_rate:.1f}% ({duplicate_rows} 行)")
              penalty += 10
          else:
              warnings.append(f"存在少量重复行: {duplicate_rows} 行")
              penalty += 5
      
      # 检查ID列的重复
      potential_id_cols = [col for col in data.columns 
                         if 'id' in col.lower() or 'key' in col.lower()]
      
      for col in potential_id_cols:
          if data[col].dtype in ['object', 'string', 'int64']:
              duplicate_ids = data[col].duplicated().sum()
              if duplicate_ids > 0:
                  issues.append(f"ID列 '{col}' 存在重复值: {duplicate_ids} 个")
                  penalty += 15
      
      return {
          'issues': issues,
          'warnings': warnings,
          'penalty': penalty
      }
  
  def _check_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
      """检查数据一致性"""
      issues = []
      warnings = []
      penalty = 0
      
      # 检查分类变量的一致性
      categorical_cols = data.select_dtypes(include=['object', 'category']).columns
      
      for col in categorical_cols:
          unique_values = data[col].dropna().unique()
          
          # 检查是否有相似但不同的值（可能的输入错误）
          if len(unique_values) > 1:
              similar_pairs = self._find_similar_values(unique_values)
              for pair in similar_pairs:
                  warnings.append(
                      f"列 '{col}' 中可能存在输入错误: '{pair[0]}' 和 '{pair[1]}'"
                  )
                  penalty += 3
      
      # 检查数值范围的合理性
      numeric_cols = data.select_dtypes(include=[np.number]).columns
      
      for col in numeric_cols:
          col_data = data[col].dropna()
          
          if len(col_data) == 0:
              continue
          
          # 检查负值（对于某些列不应该有负值）
          if col.lower() in ['age', 'weight', 'height', 'price', 'count']:
              negative_count = (col_data < 0).sum()
              if negative_count > 0:
                  issues.append(f"列 '{col}' 包含不合理的负值: {negative_count} 个")
                  penalty += 10
          
          # 检查极端值
          if col.lower() == 'age':
              extreme_age = ((col_data < 0) | (col_data > 150)).sum()
              if extreme_age > 0:
                  issues.append(f"年龄列包含不合理的值: {extreme_age} 个")
                  penalty += 10
      
      return {
          'issues': issues,
          'warnings': warnings,
          'penalty': penalty
      }
  
  def _find_similar_values(self, values: np.ndarray, threshold: float = 0.8) -> List[Tuple[str, str]]:
      """查找相似的字符串值"""
      from difflib import SequenceMatcher
      
      similar_pairs = []
      values_str = [str(v) for v in values]
      
      for i in range(len(values_str)):
          for j in range(i + 1, len(values_str)):
              similarity = SequenceMatcher(None, values_str[i], values_str[j]).ratio()
              if similarity >= threshold and values_str[i] != values_str[j]:
                  similar_pairs.append((values_str[i], values_str[j]))
      
      return similar_pairs

  def suggest_fixes(self, data: pd.DataFrame) -> Dict[str, List[str]]:
      """建议数据修复方案"""
      suggestions = {
          'missing_data': [],
          'outliers': [],
          'duplicates': [],
          'data_types': []
      }
      
      # 缺失数据处理建议
      missing_rates = (data.isnull().sum() / len(data)) * 100
      for col, rate in missing_rates.items():
          if rate > 0:
              if rate < 5:
                  suggestions['missing_data'].append(f"列 '{col}': 可以删除缺失行或使用均值/中位数填充")
              elif rate < 20:
                  suggestions['missing_data'].append(f"列 '{col}': 建议使用插值或模型预测填充")
              else:
                  suggestions['missing_data'].append(f"列 '{col}': 考虑删除该列或收集更多数据")
      
      # 异常值处理建议
      numeric_cols = data.select_dtypes(include=[np.number]).columns
      for col in numeric_cols:
          col_data = data[col].dropna()
          if len(col_data) >= 10:
              Q1, Q3 = col_data.quantile([0.25, 0.75])
              IQR = Q3 - Q1
              outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
              if len(outliers) > 0:
                  suggestions['outliers'].append(
                      f"列 '{col}': 检查 {len(outliers)} 个异常值，考虑删除或转换"
                  )
      
      return suggestions
