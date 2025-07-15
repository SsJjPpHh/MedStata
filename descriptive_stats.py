import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.statistical_methods import DescriptiveStatistics
from utils.visualization_utils import create_distribution_plot

def show():
  """显示描述性统计页面"""
  st.markdown('<h1 class="main-header">📈 描述性统计分析</h1>', unsafe_allow_html=True)
  
  if st.session_state.data is None:
      st.warning("⚠️ 请先在数据导入页面加载数据")
      return
  
  data = st.session_state.data
  
  # 创建标签页
  tab1, tab2, tab3, tab4 = st.tabs(["基础统计", "分布分析", "相关分析", "分组统计"])
  
  with tab1:
      show_basic_statistics(data)
  
  with tab2:
      show_distribution_analysis(data)
  
  with tab3:
      show_correlation_analysis(data)
  
  with tab4:
      show_group_statistics(data)

def show_basic_statistics(data: pd.DataFrame):
  """显示基础统计信息"""
  st.markdown('<h2 class="sub-header">基础统计信息</h2>', unsafe_allow_html=True)
  
  # 选择要分析的列
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
  
  if not numeric_cols and not categorical_cols:
      st.error("数据中没有可分析的列")
      return
  
  # 数值变量统计
  if numeric_cols:
      st.markdown("### 📊 数值变量统计")
      
      selected_numeric = st.multiselect(
          "选择数值变量",
          numeric_cols,
          default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
      )
      
      if selected_numeric:
          stats_calculator = DescriptiveStatistics()
          numeric_stats = stats_calculator.calculate_numeric_statistics(data[selected_numeric])
          
          # 显示统计表格
          st.dataframe(numeric_stats, use_container_width=True)
          
          # 创建统计指标卡片
          create_metric_cards(data[selected_numeric])
  
  # 分类变量统计
  if categorical_cols:
      st.markdown("### 📋 分类变量统计")
      
      selected_categorical = st.multiselect(
          "选择分类变量",
          categorical_cols,
          default=categorical_cols[:3] if len(categorical_cols) > 3 else categorical_cols
      )
      
      if selected_categorical:
          for col in selected_categorical:
              show_categorical_summary(data, col)

def create_metric_cards(data: pd.DataFrame):
  """创建统计指标卡片"""
  cols = st.columns(4)
  
  for i, col in enumerate(data.columns[:4]):
      with cols[i % 4]:
          col_data = data[col].dropna()
          
          st.metric(
              label=f"{col} - 均值",
              value=f"{col_data.mean():.2f}",
              delta=f"标准差: {col_data.std():.2f}"
          )

def show_categorical_summary(data: pd.DataFrame, column: str):
  """显示分类变量摘要"""
  st.markdown(f"#### {column}")
  
  col1, col2 = st.columns([1, 2])
  
  with col1:
      # 频数统计
      value_counts = data[column].value_counts()
      freq_df = pd.DataFrame({
          '类别': value_counts.index,
          '频数': value_counts.values,
          '频率(%)': (value_counts.values / len(data) * 100).round(2)
      })
      st.dataframe(freq_df, use_container_width=True)
  
  with col2:
      # 饼图
      fig = px.pie(
          values=value_counts.values,
          names=value_counts.index,
          title=f"{column} 分布"
      )
      st.plotly_chart(fig, use_container_width=True)

def show_distribution_analysis(data: pd.DataFrame):
  """显示分布分析"""
  st.markdown('<h2 class="sub-header">分布分析</h2>', unsafe_allow_html=True)
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if not numeric_cols:
      st.warning("没有数值变量可以进行分布分析")
      return
  
  # 选择变量
  selected_var = st.selectbox("选择要分析的变量", numeric_cols)
  
  if selected_var:
      col_data = data[selected_var].dropna()
      
      # 创建分布图
      col1, col2 = st.columns(2)
      
      with col1:
          # 直方图
          fig_hist = px.histogram(
              x=col_data,
              nbins=30,
              title=f"{selected_var} - 直方图",
              labels={'x': selected_var, 'y': '频数'}
          )
          st.plotly_chart(fig_hist, use_container_width=True)
      
      with col2:
          # 箱线图
          fig_box = px.box(
              y=col_data,
              title=f"{selected_var} - 箱线图"
          )
          st.plotly_chart(fig_box, use_container_width=True)
      
      # 正态性检验
      st.markdown("### 正态性检验")
      
      from scipy import stats
      
      # Shapiro-Wilk检验
      if len(col_data) <= 5000:  # Shapiro-Wilk检验的样本量限制
          shapiro_stat, shapiro_p = stats.shapiro(col_data)
          
          col1, col2, col3 = st.columns(3)
          
          with col1:
              st.metric("Shapiro-Wilk统计量", f"{shapiro_stat:.4f}")
          with col2:
              st.metric("p值", f"{shapiro_p:.4f}")
          with col3:
              if shapiro_p > 0.05:
                  st.success("✅ 符合正态分布")
              else:
                  st.error("❌ 不符合正态分布")
      
      # 描述性统计
      st.markdown("### 分布特征")
      
      col1, col2, col3, col4 = st.columns(4)
      
      with col1:
          st.metric("偏度", f"{stats.skew(col_data):.3f}")
      with col2:
          st.metric("峰度", f"{stats.kurtosis(col_data):.3f}")
      with col3:
          st.metric("变异系数", f"{(col_data.std() / col_data.mean()):.3f}")
      with col4:
          st.metric("四分位距", f"{col_data.quantile(0.75) - col_data.quantile(0.25):.3f}")

def show_correlation_analysis(data: pd.DataFrame):
  """显示相关分析"""
  st.markdown('<h2 class="sub-header">相关分析</h2>', unsafe_allow_html=True)
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if len(numeric_cols) < 2:
      st.warning("需要至少2个数值变量进行相关分析")
      return
  
  # 选择相关系数类型
  corr_method = st.selectbox(
      "选择相关系数类型",
      ["pearson", "spearman", "kendall"],
      format_func=lambda x: {
          "pearson": "皮尔逊相关系数",
          "spearman": "斯皮尔曼等级相关",
          "kendall": "肯德尔τ相关"
      }[x]
  )
  
  # 计算相关矩阵
  corr_matrix = data[numeric_cols].corr(method=corr_method)
  
  # 相关矩阵热图
  fig = px.imshow(
      corr_matrix,
      text_auto=True,
      aspect="auto",
      title=f"相关矩阵热图 ({corr_method})",
      color_continuous_scale="RdBu_r",
      zmin=-1, zmax=1
  )
  
  st.plotly_chart(fig, use_container_width=True)
  
  # 显示相关系数表格
  st.markdown("### 相关系数矩阵")
  st.dataframe(corr_matrix.round(3), use_container_width=True)
  
  # 强相关关系识别
  st.markdown("### 强相关关系识别")
  
  threshold = st.slider("相关系数阈值", 0.5, 0.9, 0.7, 0.05)
  
  strong_correlations = []
  for i in range(len(corr_matrix.columns)):
      for j in range(i+1, len(corr_matrix.columns)):
          corr_val = corr_matrix.iloc[i, j]
          if abs(corr_val) >= threshold:
              strong_correlations.append({
                  '变量1': corr_matrix.columns[i],
                  '变量2': corr_matrix.columns[j],
                  '相关系数': corr_val,
                  '相关强度': get_correlation_strength(abs(corr_val))
              })
  
  if strong_correlations:
      strong_corr_df = pd.DataFrame(strong_correlations)
      st.dataframe(strong_corr_df, use_container_width=True)
  else:
      st.info(f"没有发现相关系数绝对值大于 {threshold} 的变量对")

def show_group_statistics(data: pd.DataFrame):
  """显示分组统计"""
  st.markdown('<h2 class="sub-header">分组统计分析</h2>', unsafe_allow_html=True)
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
  
  if not numeric_cols or not categorical_cols:
      st.warning("需要至少一个数值变量和一个分类变量进行分组分析")
      return
  
  # 选择分组变量和分析变量
  col1, col2 = st.columns(2)
  
  with col1:
      group_var = st.selectbox("选择分组变量", categorical_cols)
  
  with col2:
      analysis_var = st.selectbox("选择分析变量", numeric_cols)
  
  if group_var and analysis_var:
      # 分组统计
      grouped_stats = data.groupby(group_var)[analysis_var].agg([
          'count', 'mean', 'std', 'min', 'max', 'median'
      ]).round(3)
      
      grouped_stats.columns = ['样本量', '均值', '标准差', '最小值', '最大值', '中位数']
      
      st.markdown("### 分组统计表")
      st.dataframe(grouped_stats, use_container_width=True)
      
      # 可视化
      col1, col2 = st.columns(2)
      
      with col1:
          # 箱线图
          fig_box = px.box(
              data, 
              x=group_var, 
              y=analysis_var,
              title=f"{analysis_var} 按 {group_var} 分组的箱线图"
          )
          st.plotly_chart(fig_box, use_container_width=True)
      
      with col2:
          # 小提琴图
          fig_violin = px.violin(
              data, 
              x=group_var, 
              y=analysis_var,
              title=f"{analysis_var} 按 {group_var} 分组的小提琴图"
          )
          st.plotly_chart(fig_violin, use_container_width=True)
      
      # 统计检验
      st.markdown("### 统计检验")
      
      groups = [group[analysis_var].dropna() for name, group in data.groupby(group_var)]
      
      if len(groups) == 2:
          # 两组比较：t检验
          from scipy.stats import ttest_ind, mannwhitneyu
          
          # t检验
          t_stat, t_p = ttest_ind(groups[0], groups[1])
          
          # Mann-Whitney U检验
          u_stat, u_p = mannwhitneyu(groups[0], groups[1])
          
          col1, col2 = st.columns(2)
          
          with col1:
              st.markdown("**独立样本t检验**")
              st.write(f"t统计量: {t_stat:.4f}")
              st.write(f"p值: {t_p:.4f}")
              if t_p < 0.05:
                  st.success("✅ 差异显著 (p < 0.05)")
              else:
                  st.info("ℹ️ 差异不显著 (p ≥ 0.05)")
          
          with col2:
              st.markdown("**Mann-Whitney U检验**")
              st.write(f"U统计量: {u_stat:.4f}")
              st.write(f"p值: {u_p:.4f}")
              if u_p < 0.05:
                  st.success("✅ 差异显著 (p < 0.05)")
              else:
                  st.info("ℹ️ 差异不显著 (p ≥ 0.05)")
      
      elif len(groups) > 2:
          # 多组比较：方差分析
          from scipy.stats import f_oneway, kruskal
          
          # 单因素方差分析
          f_stat, f_p = f_oneway(*groups)
          
          # Kruskal-Wallis检验
          h_stat, h_p = kruskal(*groups)
          
          col1, col2 = st.columns(2)
          
          with col1:
              st.markdown("**单因素方差分析**")
              st.write(f"F统计量: {f_stat:.4f}")
              st.write(f"p值: {f_p:.4f}")
              if f_p < 0.05:
                  st.success("✅ 组间差异显著 (p < 0.05)")
              else:
                  st.info("ℹ️ 组间差异不显著 (p ≥ 0.05)")
          
          with col2:
              st.markdown("**Kruskal-Wallis检验**")
              st.write(f"H统计量: {h_stat:.4f}")
              st.write(f"p值: {h_p:.4f}")
              if h_p < 0.05:
                  st.success("✅ 组间差异显著 (p < 0.05)")
              else:
                  st.info("ℹ️ 组间差异不显著 (p ≥ 0.05)")

def get_correlation_strength(corr_val: float) -> str:
  """获取相关强度描述"""
  if corr_val >= 0.8:
      return "很强"
  elif corr_val >= 0.6:
      return "强"
  elif corr_val >= 0.4:
      return "中等"
  elif corr_val >= 0.2:
      return "弱"
  else:
      return "很弱"
