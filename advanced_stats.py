import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.statistical_methods import InferentialStatistics, NonParametricTests, EffectSizeCalculator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report
import seaborn as sns

def show():
  """显示高级统计分析页面"""
  st.markdown('<h1 class="main-header">🔬 高级统计分析</h1>', unsafe_allow_html=True)
  
  if st.session_state.data is None:
      st.warning("⚠️ 请先在数据导入页面加载数据")
      return
  
  data = st.session_state.data
  
  # 创建标签页
  tab1, tab2, tab3, tab4, tab5 = st.tabs([
      "假设检验", "回归分析", "非参数检验", "效应量分析", "多变量分析"
  ])
  
  with tab1:
      show_hypothesis_testing(data)
  
  with tab2:
      show_regression_analysis(data)
  
  with tab3:
      show_nonparametric_tests(data)
  
  with tab4:
      show_effect_size_analysis(data)
  
  with tab5:
      show_multivariate_analysis(data)

def show_hypothesis_testing(data: pd.DataFrame):
  """显示假设检验"""
  st.markdown('<h2 class="sub-header">假设检验</h2>', unsafe_allow_html=True)
  
  test_type = st.selectbox(
      "选择检验类型",
      ["单样本t检验", "独立样本t检验", "配对样本t检验", "卡方检验", "单因素方差分析"]
  )
  
  inferential_stats = InferentialStatistics()
  
  if test_type == "单样本t检验":
      show_one_sample_ttest(data, inferential_stats)
  elif test_type == "独立样本t检验":
      show_two_sample_ttest(data, inferential_stats)
  elif test_type == "配对样本t检验":
      show_paired_ttest(data, inferential_stats)
  elif test_type == "卡方检验":
      show_chi_square_test(data, inferential_stats)
  elif test_type == "单因素方差分析":
      show_anova_test(data, inferential_stats)

def show_one_sample_ttest(data: pd.DataFrame, inferential_stats: InferentialStatistics):
  """单样本t检验"""
  st.markdown("### 单样本t检验")
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if not numeric_cols:
      st.warning("没有数值变量可进行检验")
      return
  
  col1, col2 = st.columns(2)
  
  with col1:
      test_var = st.selectbox("选择检验变量", numeric_cols)
  
  with col2:
      test_value = st.number_input("检验值（μ₀）", value=0.0)
  
  if st.button("执行检验"):
      result = inferential_stats.one_sample_ttest(data[test_var], test_value)
      
      # 显示结果
      col1, col2, col3 = st.columns(3)
      
      with col1:
          st.metric("t统计量", f"{result['statistic']:.4f}")
      with col2:
          st.metric("p值", f"{result['p_value']:.4f}")
      with col3:
          st.metric("自由度", result['degrees_of_freedom'])
      
      # 结果解释
      if result['significant']:
          st.success(f"✅ {result['interpretation']}")
      else:
          st.info(f"ℹ️ {result['interpretation']}")
      
      # 可视化
      fig = create_ttest_visualization(data[test_var], test_value, result)
      st.plotly_chart(fig, use_container_width=True)

def show_two_sample_ttest(data: pd.DataFrame, inferential_stats: InferentialStatistics):
  """独立样本t检验"""
  st.markdown("### 独立样本t检验")
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
  
  if not numeric_cols or not categorical_cols:
      st.warning("需要至少一个数值变量和一个分类变量")
      return
  
  col1, col2 = st.columns(2)
  
  with col1:
      test_var = st.selectbox("选择检验变量", numeric_cols)
  
  with col2:
      group_var = st.selectbox("选择分组变量", categorical_cols)
  
  # 检查分组变量的唯一值
  unique_groups = data[group_var].dropna().unique()
  
  if len(unique_groups) != 2:
      st.warning(f"分组变量必须恰好有2个类别，当前有{len(unique_groups)}个")
      return
  
  equal_var = st.checkbox("假设等方差", value=True)
  
  if st.button("执行检验"):
      group1_data = data[data[group_var] == unique_groups[0]][test_var]
      group2_data = data[data[group_var] == unique_groups[1]][test_var]
      
      result = inferential_stats.two_sample_ttest(group1_data, group2_data, equal_var)
      
      # 显示结果
      col1, col2 = st.columns(2)
      
      with col1:
          st.metric("t统计量", f"{result['statistic']:.4f}")
          st.metric(f"{unique_groups[0]} 均值", f"{result['group1_mean']:.4f}")
          st.metric(f"{unique_groups[0]} 标准差", f"{result['group1_std']:.4f}")
      
      with col2:
          st.metric("p值", f"{result['p_value']:.4f}")
          st.metric(f"{unique_groups[1]} 均值", f"{result['group2_mean']:.4f}")
          st.metric(f"{unique_groups[1]} 标准差", f"{result['group2_std']:.4f}")
      
      # 结果解释
      if result['significant']:
          st.success(f"✅ {result['interpretation']}")
      else:
          st.info(f"ℹ️ {result['interpretation']}")
      
      # 效应量
      effect_size = EffectSizeCalculator.cohens_d(group1_data, group2_data)
      st.metric("Cohen's d", f"{effect_size:.4f}")
      st.info(f"效应量: {EffectSizeCalculator.interpret_cohens_d(effect_size)}")

def show_regression_analysis(data: pd.DataFrame):
  """显示回归分析"""
  st.markdown('<h2 class="sub-header">回归分析</h2>', unsafe_allow_html=True)
  
  regression_type = st.selectbox(
      "选择回归类型",
      ["线性回归", "多元线性回归", "逻辑回归"]
  )
  
  if regression_type == "线性回归":
      show_simple_linear_regression(data)
  elif regression_type == "多元线性回归":
      show_multiple_linear_regression(data)
  elif regression_type == "逻辑回归":
      show_logistic_regression(data)

def show_simple_linear_regression(data: pd.DataFrame):
  """简单线性回归"""
  st.markdown("### 简单线性回归")
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if len(numeric_cols) < 2:
      st.warning("需要至少2个数值变量进行回归分析")
      return
  
  col1, col2 = st.columns(2)
  
  with col1:
      y_var = st.selectbox("选择因变量 (Y)", numeric_cols)
  
  with col2:
      x_var = st.selectbox("选择自变量 (X)", [col for col in numeric_cols if col != y_var])
  
  if st.button("执行回归分析"):
      # 准备数据
      mask = ~(data[x_var].isna() | data[y_var].isna())
      X = data[x_var][mask].values.reshape(-1, 1)
      y = data[y_var][mask].values
      
      # 拟合模型
      model = LinearRegression()
      model.fit(X, y)
      
      # 预测
      y_pred = model.predict(X)
      
      # 计算统计量
      r2 = r2_score(y, y_pred)
      correlation = np.corrcoef(X.flatten(), y)[0, 1]
      
      # 显示结果
      col1, col2, col3 = st.columns(3)
      
      with col1:
          st.metric("R²", f"{r2:.4f}")
      with col2:
          st.metric("相关系数", f"{correlation:.4f}")
      with col3:
          st.metric("回归系数", f"{model.coef_[0]:.4f}")
      
      st.metric("截距", f"{model.intercept_:.4f}")
      
      # 回归方程
      st.markdown("### 回归方程")
      equation = f"{y_var} = {model.intercept_:.4f} + {model.coef_[0]:.4f} × {x_var}"
      st.latex(equation.replace('×', r'\times'))
      
      # 可视化
      fig = px.scatter(data, x=x_var, y=y_var, title=f"{y_var} vs {x_var}")
      
      # 添加回归线
      x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
      y_range = model.predict(x_range)
      
      fig.add_scatter(
          x=x_range.flatten(),
          y=y_range,
          mode='lines',
          name='回归线',
          line=dict(color='red')
      )
      
      st.plotly_chart(fig, use_container_width=True)

def show_multiple_linear_regression(data: pd.DataFrame):
  """多元线性回归"""
  st.markdown("### 多元线性回归")
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if len(numeric_cols) < 3:
      st.warning("需要至少3个数值变量进行多元回归分析")
      return
  
  y_var = st.selectbox("选择因变量 (Y)", numeric_cols)
  x_vars = st.multiselect(
      "选择自变量 (X)",
      [col for col in numeric_cols if col != y_var],
      default=[col for col in numeric_cols if col != y_var][:3]
  )
  
  if len(x_vars) < 1:
      st.warning("请至少选择一个自变量")
      return
  
  if st.button("执行多元回归分析"):
      # 准备数据
      vars_to_use = [y_var] + x_vars
      clean_data = data[vars_to_use].dropna()
      
      X = clean_data[x_vars].values
      y = clean_data[y_var].values
      
      # 拟合模型
      model = LinearRegression()
      model.fit(X, y)
      
      # 预测
      y_pred = model.predict(X)
      
      # 计算统计量
      r2 = r2_score(y, y_pred)
      adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(x_vars) - 1)
      
      # 显示结果
      col1, col2, col3 = st.columns(3)
      
      with col1:
          st.metric("R²", f"{r2:.4f}")
      with col2:
          st.metric("调整R²", f"{adj_r2:.4f}")
      with col3:
          st.metric("样本量", len(y))
      
      # 回归系数表
      st.markdown("### 回归系数")
      
      coef_df = pd.DataFrame({
          '变量': ['截距'] + x_vars,
          '系数': [model.intercept_] + model.coef_.tolist(),
          '标准化系数': [0] + (model.coef_ * np.std(X, axis=0) / np.std(y)).tolist()
      })
      
      st.dataframe(coef_df.round(4), use_container_width=True)
      
      # 残差分析
      residuals = y - y_pred
      
      col1, col2 = st.columns(2)
      
      with col1:
          # 残差vs拟合值图
          fig_resid = px.scatter(
              x=y_pred, y=residuals,
              title="残差 vs 拟合值",
              labels={'x': '拟合值', 'y': '残差'}
          )
          fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
          st.plotly_chart(fig_resid, use_container_width=True)
      
      with col2:
          # 残差直方图
          fig_hist = px.histogram(
              x=residuals,
              title="残差分布",
              labels={'x': '残差', 'y': '频数'}
          )
          st.plotly_chart(fig_hist, use_container_width=True)

def show_nonparametric_tests(data: pd.DataFrame):
  """显示非参数检验"""
  st.markdown('<h2 class="sub-header">非参数检验</h2>', unsafe_allow_html=True)
  
  test_type = st.selectbox(
      "选择非参数检验类型",
      ["Mann-Whitney U检验", "Wilcoxon符号秩检验", "Kruskal-Wallis检验"]
  )
  
  nonparam_tests = NonParametricTests()
  
  if test_type == "Mann-Whitney U检验":
      show_mann_whitney_test(data, nonparam_tests)
  elif test_type == "Wilcoxon符号秩检验":
      show_wilcoxon_test(data, nonparam_tests)
  elif test_type == "Kruskal-Wallis检验":
      show_kruskal_wallis_test(data, nonparam_tests)

def show_mann_whitney_test(data: pd.DataFrame, nonparam_tests: NonParametricTests):
  """Mann-Whitney U检验"""
  st.markdown("### Mann-Whitney U检验")
  st.info("💡 用于比较两个独立组的中位数差异（非参数版本的独立样本t检验）")
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
  
  if not numeric_cols or not categorical_cols:
      st.warning("需要至少一个数值变量和一个分类变量")
      return
  
  col1, col2 = st.columns(2)
  
  with col1:
      test_var = st.selectbox("选择检验变量", numeric_cols)
  
  with col2:
      group_var = st.selectbox("选择分组变量", categorical_cols)
  
  # 检查分组变量的唯一值
  unique_groups = data[group_var].dropna().unique()
  
  if len(unique_groups) != 2:
      st.warning(f"分组变量必须恰好有2个类别，当前有{len(unique_groups)}个")
      return
  
  if st.button("执行Mann-Whitney U检验"):
      group1_data = data[data[group_var] == unique_groups[0]][test_var]
      group2_data = data[data[group_var] == unique_groups[1]][test_var]
      
      result = nonparam_tests.mann_whitney_u_test(group1_data, group2_data)
      
      # 显示结果
      col1, col2, col3 = st.columns(3)
      
      with col1:
          st.metric("U统计量", f"{result['u_statistic']:.0f}")
      with col2:
          st.metric("p值", f"{result['p_value']:.4f}")
      with col3:
          if result['significant']:
              st.success("✅ 显著")
          else:
              st.info("ℹ️ 不显著")
      
      # 组别统计
      st.markdown("### 组别统计")
      
      group_stats = pd.DataFrame({
          '组别': [unique_groups[0], unique_groups[1]],
          '样本量': [result['group1_size'], result['group2_size']],
          '中位数': [result['group1_median'], result['group2_median']]
      })
      
      st.dataframe(group_stats, use_container_width=True)
      
      # 结果解释
      st.markdown("### 检验结果")
      st.write(result['interpretation'])
      
      # 可视化
      fig = create_group_comparison_plot(data, test_var, group_var, "Mann-Whitney U检验结果")
      st.plotly_chart(fig, use_container_width=True)

def show_wilcoxon_test(data: pd.DataFrame, nonparam_tests: NonParametricTests):
  """Wilcoxon符号秩检验"""
  st.markdown("### Wilcoxon符号秩检验")
  st.info("💡 用于比较配对样本的中位数差异（非参数版本的配对t检验）")
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if len(numeric_cols) < 2:
      st.warning("需要至少2个数值变量进行配对检验")
      return
  
  col1, col2 = st.columns(2)
  
  with col1:
      before_var = st.selectbox("选择前测变量", numeric_cols)
  
  with col2:
      after_var = st.selectbox("选择后测变量", [col for col in numeric_cols if col != before_var])
  
  if st.button("执行Wilcoxon符号秩检验"):
      result = nonparam_tests.wilcoxon_signed_rank_test(data[before_var], data[after_var])
      
      # 显示结果
      col1, col2, col3 = st.columns(3)
      
      with col1:
          st.metric("W统计量", f"{result['w_statistic']:.0f}")
      with col2:
          st.metric("p值", f"{result['p_value']:.4f}")
      with col3:
          st.metric("样本量", result['sample_size'])
      
      # 前后测统计
      col1, col2 = st.columns(2)
      
      with col1:
          st.metric(f"{before_var} 中位数", f"{result['before_median']:.4f}")
      with col2:
          st.metric(f"{after_var} 中位数", f"{result['after_median']:.4f}")
      
      # 结果解释
      if result['significant']:
          st.success(f"✅ {result['interpretation']}")
      else:
          st.info(f"ℹ️ {result['interpretation']}")
      
      # 差值分析
      differences = data[after_var] - data[before_var]
      differences_clean = differences.dropna()
      
      st.markdown("### 差值分析")
      
      col1, col2, col3 = st.columns(3)
      
      with col1:
          st.metric("改善例数", (differences_clean > 0).sum())
      with col2:
          st.metric("恶化例数", (differences_clean < 0).sum())
      with col3:
          st.metric("无变化例数", (differences_clean == 0).sum())

def show_effect_size_analysis(data: pd.DataFrame):
  """显示效应量分析"""
  st.markdown('<h2 class="sub-header">效应量分析</h2>', unsafe_allow_html=True)
  
  st.info("💡 效应量用于衡量统计显著性的实际意义大小")
  
  effect_type = st.selectbox(
      "选择效应量类型",
      ["Cohen's d", "相关系数的决定系数", "Eta平方"]
  )
  
  if effect_type == "Cohen's d":
      show_cohens_d_analysis(data)
  elif effect_type == "相关系数的决定系数":
      show_r_squared_analysis(data)
  elif effect_type == "Eta平方":
      show_eta_squared_analysis(data)

def show_cohens_d_analysis(data: pd.DataFrame):
  """Cohen's d分析"""
  st.markdown("### Cohen's d 效应量分析")
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
  
  if not numeric_cols or not categorical_cols:
      st.warning("需要至少一个数值变量和一个分类变量")
      return
  
  col1, col2 = st.columns(2)
  
  with col1:
      test_var = st.selectbox("选择检验变量", numeric_cols, key="cohens_d_var")
  
  with col2:
      group_var = st.selectbox("选择分组变量", categorical_cols, key="cohens_d_group")
  
  unique_groups = data[group_var].dropna().unique()
  
  if len(unique_groups) != 2:
      st.warning(f"分组变量必须恰好有2个类别")
      return
  
  if st.button("计算Cohen's d"):
      group1_data = data[data[group_var] == unique_groups[0]][test_var].dropna()
      group2_data = data[data[group_var] == unique_groups[1]][test_var].dropna()
      
      cohens_d = EffectSizeCalculator.cohens_d(group1_data, group2_data)
      interpretation = EffectSizeCalculator.interpret_cohens_d(cohens_d)
      
      # 显示结果
      col1, col2 = st.columns(2)
      
      with col1:
          st.metric("Cohen's d", f"{cohens_d:.4f}")
      with col2:
          st.info(f"效应量大小: {interpretation}")
      
      # 效应量解释表
      st.markdown("### Cohen's d 解释标准")
      
      interpretation_df = pd.DataFrame({
          '效应量范围': ['|d| < 0.2', '0.2 ≤ |d| < 0.5', '0.5 ≤ |d| < 0.8', '|d| ≥ 0.8'],
          '效应大小': ['无效应或极小', '小效应', '中等效应', '大效应'],
          '实际意义': ['几乎无差异', '较小差异', '中等差异', '很大差异']
      })
      
      st.dataframe(interpretation_df, use_container_width=True)

def show_multivariate_analysis(data: pd.DataFrame):
  """显示多变量分析"""
  st.markdown('<h2 class="sub-header">多变量分析</h2>', unsafe_allow_html=True)
  
  analysis_type = st.selectbox(
      "选择分析类型",
      ["主成分分析(PCA)", "因子分析", "聚类分析", "判别分析"]
  )
  
  if analysis_type == "主成分分析(PCA)":
      show_pca_analysis(data)
  elif analysis_type == "因子分析":
      show_factor_analysis(data)
  elif analysis_type == "聚类分析":
      show_cluster_analysis(data)
  elif analysis_type == "判别分析":
      show_discriminant_analysis(data)

def show_pca_analysis(data: pd.DataFrame):
  """主成分分析"""
  st.markdown("### 主成分分析 (PCA)")
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if len(numeric_cols) < 3:
      st.warning("需要至少3个数值变量进行PCA分析")
      return
  
  selected_vars = st.multiselect(
      "选择用于PCA的变量",
      numeric_cols,
      default=numeric_cols
  )
  
  if len(selected_vars) < 3:
      st.warning("请至少选择3个变量")
      return
  
  standardize = st.checkbox("标准化数据", value=True)
  
  if st.button("执行PCA分析"):
      from sklearn.decomposition import PCA
      from sklearn.preprocessing import StandardScaler
      
      # 准备数据
      pca_data = data[selected_vars].dropna()
      
      if standardize:
          scaler = StandardScaler()
          pca_data_scaled = scaler.fit_transform(pca_data)
      else:
          pca_data_scaled = pca_data.values
      
      # 执行PCA
      pca = PCA()
      pca_result = pca.fit_transform(pca_data_scaled)
      
      # 方差解释比例
      explained_variance_ratio = pca.explained_variance_ratio_
      cumulative_variance = np.cumsum(explained_variance_ratio)
      
      # 显示结果
      st.markdown("### PCA结果")
      
      # 方差解释表
      pca_summary = pd.DataFrame({
          '主成分': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
          '特征值': pca.explained_variance_,
          '方差解释比例': explained_variance_ratio,
          '累积方差解释比例': cumulative_variance
      })
      
      st.dataframe(pca_summary.round(4), use_container_width=True)
      
      # 碎石图
      fig_scree = px.line(
          x=range(1, len(explained_variance_ratio) + 1),
          y=explained_variance_ratio,
          title="碎石图",
          labels={'x': '主成分', 'y': '方差解释比例'}
      )
      fig_scree.add_scatter(
          x=list(range(1, len(explained_variance_ratio) + 1)),
          y=explained_variance_ratio,
          mode='markers',
          marker=dict(size=8)
      )
      st.plotly_chart(fig_scree, use_container_width=True)
      
      # 主成分载荷矩阵
      st.markdown("### 主成分载荷矩阵")
      
      n_components_show = min(5, len(selected_vars))
      loadings = pd.DataFrame(
          pca.components_[:n_components_show].T,
          columns=[f'PC{i+1}' for i in range(n_components_show)],
          index=selected_vars
      )
      
      st.dataframe(loadings.round(4), use_container_width=True)
      
      # 双标图（前两个主成分）
      if len(pca_result) > 0:
          fig_biplot = px.scatter(
              x=pca_result[:, 0],
              y=pca_result[:, 1],
              title="PCA双标图 (PC1 vs PC2)",
              labels={'x': f'PC1 ({explained_variance_ratio[0]:.1%})', 
                     'y': f'PC2 ({explained_variance_ratio[1]:.1%})'}
          )
          st.plotly_chart(fig_biplot, use_container_width=True)

def create_group_comparison_plot(data: pd.DataFrame, y_var: str, group_var: str, title: str):
  """创建组间比较图"""
  fig = make_subplots(
      rows=1, cols=2,
      subplot_titles=["箱线图", "小提琴图"]
  )
  
  # 箱线图
  for group in data[group_var].unique():
      if pd.notna(group):
          group_data = data[data[group_var] == group][y_var].dropna()
          fig.add_trace(
              go.Box(y=group_data, name=str(group), showlegend=False),
              row=1, col=1
          )
  
  # 小提琴图
  for group in data[group_var].unique():
      if pd.notna(group):
          group_data = data[data[group_var] == group][y_var].dropna()
          fig.add_trace(
              go.Violin(y=group_data, name=str(group), showlegend=False),
              row=1, col=2
          )
  
  fig.update_layout(title_text=title, showlegend=False)
  fig.update_yaxes(title_text=y_var)
  
  return fig

def create_ttest_visualization(data: pd.Series, test_value: float, result: dict):
  """创建t检验可视化"""
  fig = make_subplots(
      rows=1, cols=2,
      subplot_titles=["数据分布", "t统计量分布"]
  )
  
  # 数据分布直方图
  fig.add_trace(
      go.Histogram(x=data.dropna(), name="数据分布", showlegend=False),
      row=1, col=1
  )
  
  # 添加均值和检验值线
  fig.add_vline(x=data.mean(), line_dash="dash", line_color="red", 
                annotation_text=f"样本均值: {data.mean():.2f}", row=1, col=1)
  fig.add_vline(x=test_value, line_dash="dash", line_color="blue",
                annotation_text=f"检验值: {test_value:.2f}", row=1, col=1)
  
  # t分布
  from scipy.stats import t
  df = result['degrees_of_freedom']
  x_range = np.linspace(-4, 4, 1000)
  t_dist = t.pdf(x_range, df)
  
  fig.add_trace(
      go.Scatter(x=x_range, y=t_dist, name="t分布", showlegend=False),
      row=1, col=2
  )
  
  # 添加t统计量线
  fig.add_vline(x=result['statistic'], line_dash="dash", line_color="red",
                annotation_text=f"t = {result['statistic']:.3f}", row=1, col=2)
  
  fig.update_layout(title_text="单样本t检验可视化")
  
  return fig
