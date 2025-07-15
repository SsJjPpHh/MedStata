import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple

class MedicalPlotGenerator:
  """医学图表生成器"""
  
  def __init__(self):
      self.color_palette = [
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf'
      ]
  
  def create_distribution_plot(self, data: pd.Series, title: str = None, 
                             plot_type: str = 'histogram') -> go.Figure:
      """创建分布图"""
      if title is None:
          title = f"{data.name} 分布"
      
      if plot_type == 'histogram':
          fig = px.histogram(
              x=data.dropna(),
              title=title,
              nbins=30,
              labels={'x': data.name, 'y': '频数'}
          )
      elif plot_type == 'box':
          fig = px.box(
              y=data.dropna(),
              title=title,
              labels={'y': data.name}
          )
      elif plot_type == 'violin':
          fig = px.violin(
              y=data.dropna(),
              title=title,
              labels={'y': data.name}
          )
      else:
          raise ValueError("plot_type must be 'histogram', 'box', or 'violin'")
      
      return fig
  
  def create_correlation_heatmap(self, data: pd.DataFrame, 
                               method: str = 'pearson') -> go.Figure:
      """创建相关性热图"""
      numeric_data = data.select_dtypes(include=[np.number])
      corr_matrix = numeric_data.corr(method=method)
      
      fig = px.imshow(
          corr_matrix,
          text_auto=True,
          aspect="auto",
          title=f"相关性热图 ({method})",
          color_continuous_scale="RdBu_r",
          zmin=-1, zmax=1
      )
      
      return fig
  
  def create_survival_curve(self, time_data: pd.Series, event_data: pd.Series,
                          group_data: pd.Series = None, title: str = "生存曲线") -> go.Figure:
      """创建生存曲线"""
      try:
          from lifelines import KaplanMeierFitter
      except ImportError:
          raise ImportError("需要安装lifelines库: pip install lifelines")
      
      fig = go.Figure()
      
      if group_data is None:
          # 单一生存曲线
          kmf = KaplanMeierFitter()
          kmf.fit(time_data, event_data)
          
          fig.add_trace(go.Scatter(
              x=kmf.timeline,
              y=kmf.survival_function_.iloc[:, 0],
              mode='lines',
              name='生存概率',
              line=dict(step='hv')
          ))
      else:
          # 分组生存曲线
          for group in group_data.unique():
              if pd.notna(group):
                  mask = group_data == group
                  kmf = KaplanMeierFitter()
                  kmf.fit(time_data[mask], event_data[mask], label=str(group))
                  
                  fig.add_trace(go.Scatter(
                      x=kmf.timeline,
                      y=kmf.survival_function_.iloc[:, 0],
                      mode='lines',
                      name=f'组别: {group}',
                      line=dict(step='hv')
                  ))
      
      fig.update_layout(
          title=title,
          xaxis_title="时间",
          yaxis_title="生存概率",
          yaxis=dict(range=[0, 1])
      )
      
      return fig
  
  def create_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                      title: str = "ROC曲线") -> go.Figure:
      """创建ROC曲线"""
      from sklearn.metrics import roc_curve, auc
      
      fpr, tpr, _ = roc_curve(y_true, y_scores)
      roc_auc = auc(fpr, tpr)
      
      fig = go.Figure()
      
      fig.add_trace(go.Scatter(
          x=fpr, y=tpr,
          mode='lines',
          name=f'ROC曲线 (AUC = {roc_auc:.3f})',
          line=dict(color='blue', width=2)
      ))
      
      # 添加对角线
      fig.add_trace(go.Scatter(
          x=[0, 1], y=[0, 1],
          mode='lines',
          name='随机分类器',
          line=dict(color='red', width=1, dash='dash')
      ))
      
      fig.update_layout(
          title=title,
          xaxis_title="假正率 (1-特异性)",
          yaxis_title="真正率 (敏感性)",
          xaxis=dict(range=[0, 1]),
          yaxis=dict(range=[0, 1])
      )
      
      return fig
  
    def create_forest_plot(self, data: pd.DataFrame, 
                        estimate_col: str, lower_col: str, upper_col: str,
                        label_col: str, title: str = "森林图") -> go.Figure:
      """创建森林图"""
      fig = go.Figure()
      
      # 排序数据
      data_sorted = data.sort_values(estimate_col)
      
      y_positions = list(range(len(data_sorted)))
      
      # 添加置信区间
      for i, (_, row) in enumerate(data_sorted.iterrows()):
          fig.add_trace(go.Scatter(
              x=[row[lower_col], row[upper_col]],
              y=[i, i],
              mode='lines',
              line=dict(color='gray', width=2),
              showlegend=False
          ))
      
      # 添加点估计
      fig.add_trace(go.Scatter(
          x=data_sorted[estimate_col],
          y=y_positions,
          mode='markers',
          marker=dict(color='blue', size=8, symbol='diamond'),
          name='点估计',
          text=data_sorted[label_col],
          textposition='middle right'
      ))
      
      # 添加无效应线
      fig.add_vline(x=1, line_dash="dash", line_color="red", 
                    annotation_text="无效应线")
      
      fig.update_layout(
          title=title,
          xaxis_title="效应量 (95% CI)",
          yaxis=dict(
              tickvals=y_positions,
              ticktext=data_sorted[label_col].tolist(),
              title="研究"
          ),
          height=max(400, len(data_sorted) * 50)
      )
      
      return fig
  
  def create_funnel_plot(self, effect_sizes: pd.Series, standard_errors: pd.Series,
                        title: str = "漏斗图") -> go.Figure:
      """创建漏斗图（用于检测发表偏倚）"""
      fig = go.Figure()
      
      # 散点图
      fig.add_trace(go.Scatter(
          x=effect_sizes,
          y=1/standard_errors,  # 精度（标准误的倒数）
          mode='markers',
          marker=dict(color='blue', size=6),
          name='研究'
      ))
      
      # 添加对称性参考线
      mean_effect = effect_sizes.mean()
      max_precision = (1/standard_errors).max()
      
      # 95%置信区间边界
      z_score = 1.96
      x_left = mean_effect - z_score * standard_errors
      x_right = mean_effect + z_score * standard_errors
      
      fig.add_trace(go.Scatter(
          x=[mean_effect, x_left.min()],
          y=[max_precision, 1/standard_errors.max()],
          mode='lines',
          line=dict(color='red', dash='dash'),
          name='95% CI边界',
          showlegend=False
      ))
      
      fig.add_trace(go.Scatter(
          x=[mean_effect, x_right.max()],
          y=[max_precision, 1/standard_errors.max()],
          mode='lines',
          line=dict(color='red', dash='dash'),
          showlegend=False
      ))
      
      fig.update_layout(
          title=title,
          xaxis_title="效应量",
          yaxis_title="精度 (1/SE)",
          yaxis=dict(range=[0, max_precision * 1.1])
      )
      
      return fig
  
  def create_bland_altman_plot(self, method1: pd.Series, method2: pd.Series,
                              title: str = "Bland-Altman图") -> go.Figure:
      """创建Bland-Altman一致性图"""
      # 计算差值和均值
      differences = method2 - method1
      means = (method1 + method2) / 2
      
      # 计算统计量
      mean_diff = differences.mean()
      std_diff = differences.std()
      
      # 一致性界限
      upper_loa = mean_diff + 1.96 * std_diff
      lower_loa = mean_diff - 1.96 * std_diff
      
      fig = go.Figure()
      
      # 散点图
      fig.add_trace(go.Scatter(
          x=means,
          y=differences,
          mode='markers',
          marker=dict(color='blue', size=6),
          name='观测值'
      ))
      
      # 添加参考线
      fig.add_hline(y=mean_diff, line_dash="solid", line_color="green",
                    annotation_text=f"均值差: {mean_diff:.3f}")
      fig.add_hline(y=upper_loa, line_dash="dash", line_color="red",
                    annotation_text=f"上限: {upper_loa:.3f}")
      fig.add_hline(y=lower_loa, line_dash="dash", line_color="red",
                    annotation_text=f"下限: {lower_loa:.3f}")
      
      fig.update_layout(
          title=title,
          xaxis_title="两种方法的均值",
          yaxis_title="两种方法的差值"
      )
      
      return fig
  
  def create_qq_plot(self, data: pd.Series, title: str = "Q-Q图") -> go.Figure:
      """创建Q-Q图检验正态性"""
      from scipy import stats
      
      clean_data = data.dropna()
      
      # 计算理论分位数和样本分位数
      theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(clean_data)))
      sample_quantiles = np.sort(clean_data)
      
      fig = go.Figure()
      
      # 散点图
      fig.add_trace(go.Scatter(
          x=theoretical_quantiles,
          y=sample_quantiles,
          mode='markers',
          marker=dict(color='blue', size=6),
          name='观测值'
      ))
      
      # 添加理论直线
      min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
      max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
      
      fig.add_trace(go.Scatter(
          x=[min_val, max_val],
          y=[min_val, max_val],
          mode='lines',
          line=dict(color='red', dash='dash'),
          name='理论直线'
      ))
      
      fig.update_layout(
          title=title,
          xaxis_title="理论分位数",
          yaxis_title="样本分位数"
      )
      
      return fig
  
  def create_radar_chart(self, data: pd.DataFrame, categories: List[str],
                        title: str = "雷达图") -> go.Figure:
      """创建雷达图"""
      fig = go.Figure()
      
      for i, (index, row) in enumerate(data.iterrows()):
          fig.add_trace(go.Scatterpolar(
              r=row[categories].values,
              theta=categories,
              fill='toself',
              name=str(index),
              line_color=self.color_palette[i % len(self.color_palette)]
          ))
      
      fig.update_layout(
          polar=dict(
              radialaxis=dict(
                  visible=True,
                  range=[0, data[categories].max().max()]
              )
          ),
          showlegend=True,
          title=title
      )
      
      return fig
  
  def create_sankey_diagram(self, source: List[int], target: List[int], 
                           value: List[float], labels: List[str],
                           title: str = "桑基图") -> go.Figure:
      """创建桑基图"""
      fig = go.Figure(data=[go.Sankey(
          node=dict(
              pad=15,
              thickness=20,
              line=dict(color="black", width=0.5),
              label=labels,
              color="blue"
          ),
          link=dict(
              source=source,
              target=target,
              value=value
          )
      )])
      
      fig.update_layout(title_text=title, font_size=10)
      
      return fig
  
  def create_waterfall_chart(self, categories: List[str], values: List[float],
                            title: str = "瀑布图") -> go.Figure:
      """创建瀑布图"""
      # 计算累积值
      cumulative = [0]
      for i, val in enumerate(values[:-1]):  # 除了最后一个值
          cumulative.append(cumulative[-1] + val)
      
      fig = go.Figure()
      
      # 添加柱子
      for i, (cat, val) in enumerate(zip(categories, values)):
          if i == 0:  # 起始值
              fig.add_trace(go.Bar(
                  x=[cat], y=[val],
                  name=cat,
                  marker_color='green' if val >= 0 else 'red'
              ))
          elif i == len(categories) - 1:  # 最终值
              fig.add_trace(go.Bar(
                  x=[cat], y=[cumulative[i] + val],
                  name=cat,
                  marker_color='blue'
              ))
          else:  # 中间值
              fig.add_trace(go.Bar(
                  x=[cat], y=[val],
                  base=cumulative[i],
                  name=cat,
                  marker_color='green' if val >= 0 else 'red'
              ))
      
      fig.update_layout(
          title=title,
          showlegend=False,
          xaxis_title="类别",
          yaxis_title="数值"
      )
      
      return fig

class StatisticalPlotGenerator:
  """统计图表生成器"""
  
  @staticmethod
  def create_power_analysis_plot(effect_sizes: np.ndarray, sample_sizes: np.ndarray,
                                alpha: float = 0.05, title: str = "功效分析") -> go.Figure:
      """创建功效分析图"""
      from scipy import stats
      
      fig = go.Figure()
      
      for effect_size in effect_sizes:
          powers = []
          for n in sample_sizes:
              # 计算功效（这里以t检验为例）
              delta = effect_size * np.sqrt(n/2)
              t_critical = stats.t.ppf(1 - alpha/2, df=2*n-2)
              power = 1 - stats.t.cdf(t_critical, df=2*n-2, loc=delta) + \
                     stats.t.cdf(-t_critical, df=2*n-2, loc=delta)
              powers.append(power)
          
          fig.add_trace(go.Scatter(
              x=sample_sizes,
              y=powers,
              mode='lines+markers',
              name=f'效应量 = {effect_size}',
              line=dict(width=2)
          ))
      
      # 添加功效=0.8的参考线
      fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                    annotation_text="功效 = 0.8")
      
      fig.update_layout(
          title=title,
          xaxis_title="样本量",
          yaxis_title="统计功效",
          yaxis=dict(range=[0, 1])
      )
      
      return fig
  
  @staticmethod
  def create_confidence_interval_plot(estimates: pd.Series, lower_bounds: pd.Series,
                                    upper_bounds: pd.Series, labels: pd.Series,
                                    title: str = "置信区间图") -> go.Figure:
      """创建置信区间图"""
      fig = go.Figure()
      
      y_positions = list(range(len(estimates)))
      
      # 添加置信区间
      for i, (est, lower, upper, label) in enumerate(zip(estimates, lower_bounds, upper_bounds, labels)):
          fig.add_trace(go.Scatter(
              x=[lower, upper],
              y=[i, i],
              mode='lines',
              line=dict(color='gray', width=3),
              showlegend=False
          ))
          
          # 添加点估计
          fig.add_trace(go.Scatter(
              x=[est],
              y=[i],
              mode='markers',
              marker=dict(color='blue', size=8),
              showlegend=False
          ))
      
      fig.update_layout(
          title=title,
          xaxis_title="估计值",
          yaxis=dict(
              tickvals=y_positions,
              ticktext=labels.tolist(),
              title="参数"
          ),
          height=max(400, len(estimates) * 40)
      )
      
      return fig
  
  @staticmethod
  def create_residual_plots(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "残差分析") -> go.Figure:
      """创建残差分析图"""
      residuals = y_true - y_pred
      
      fig = make_subplots(
          rows=2, cols=2,
          subplot_titles=[
              "残差 vs 拟合值",
              "残差分布",
              "Q-Q图",
              "标准化残差"
          ]
      )
      
      # 残差 vs 拟合值
      fig.add_trace(
          go.Scatter(x=y_pred, y=residuals, mode='markers', name='残差'),
          row=1, col=1
      )
      fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
      
      # 残差分布
      fig.add_trace(
          go.Histogram(x=residuals, name='残差分布', showlegend=False),
          row=1, col=2
      )
      
      # Q-Q图
      from scipy import stats
      theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
      sample_quantiles = np.sort(residuals)
      
      fig.add_trace(
          go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                    mode='markers', name='Q-Q', showlegend=False),
          row=2, col=1
      )
      
      # 添加理论直线
      min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
      max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
      fig.add_trace(
          go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', line=dict(color='red', dash='dash'),
                    name='理论线', showlegend=False),
          row=2, col=1
      )
      
      # 标准化残差
      standardized_residuals = residuals / np.std(residuals)
      fig.add_trace(
          go.Scatter(x=y_pred, y=standardized_residuals, 
                    mode='markers', name='标准化残差', showlegend=False),
          row=2, col=2
      )
      fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
      fig.add_hline(y=2, line_dash="dot", line_color="orange", row=2, col=2)
      fig.add_hline(y=-2, line_dash="dot", line_color="orange", row=2, col=2)
      
      fig.update_layout(title_text=title, showlegend=False)
      
      return fig

def create_medical_dashboard(data: pd.DataFrame, 
                         patient_id_col: str = None,
                         vital_signs_cols: List[str] = None) -> Dict[str, go.Figure]:
  """创建医学仪表板"""
  dashboard_plots = {}
  
  # 患者概况
  if patient_id_col and patient_id_col in data.columns:
      dashboard_plots['patient_overview'] = create_patient_overview(data, patient_id_col)
  
  # 生命体征趋势
  if vital_signs_cols:
      valid_cols = [col for col in vital_signs_cols if col in data.columns]
      if valid_cols:
          dashboard_plots['vital_signs'] = create_vital_signs_trend(data, valid_cols)
  
  # 数据质量概览
  dashboard_plots['data_quality'] = create_data_quality_overview(data)
  
  return dashboard_plots

def create_patient_overview(data: pd.DataFrame, patient_id_col: str) -> go.Figure:
  """创建患者概览图"""
  patient_counts = data[patient_id_col].value_counts()
  
  fig = px.histogram(
      x=patient_counts.values,
      title="患者数据记录分布",
      labels={'x': '记录数量', 'y': '患者数量'}
  )
  
  return fig

def create_vital_signs_trend(data: pd.DataFrame, vital_cols: List[str]) -> go.Figure:
  """创建生命体征趋势图"""
  fig = go.Figure()
  
  for col in vital_cols:
      if col in data.columns:
          fig.add_trace(go.Scatter(
              y=data[col].dropna(),
              mode='lines+markers',
              name=col,
              line=dict(width=2)
          ))
  
  fig.update_layout(
      title="生命体征趋势",
      xaxis_title="时间点",
      yaxis_title="数值",
      hovermode='x unified'
  )
  
  return fig

def create_data_quality_overview(data: pd.DataFrame) -> go.Figure:
  """创建数据质量概览"""
  missing_percentages = (data.isnull().sum() / len(data) * 100).sort_values(ascending=True)
  
  fig = px.bar(
      x=missing_percentages.values,
      y=missing_percentages.index,
      orientation='h',
      title="各变量缺失率",
      labels={'x': '缺失率 (%)', 'y': '变量名'},
      color=missing_percentages.values,
      color_continuous_scale='Reds'
  )
  
  return fig
