import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
  confusion_matrix, classification_report, mean_squared_error, r2_score,
  roc_curve, precision_recall_curve
)

# 分类算法
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# 回归算法
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 聚类算法
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def show():
  """显示机器学习页面"""
  st.markdown('<h1 class="main-header">🤖 机器学习分析</h1>', unsafe_allow_html=True)
  
  if st.session_state.data is None:
      st.warning("⚠️ 请先在数据导入页面加载数据")
      return
  
  data = st.session_state.data
  
  # 创建标签页
  tab1, tab2, tab3, tab4 = st.tabs([
      "分类算法", "回归算法", "聚类分析", "模型评估"
  ])
  
  with tab1:
      show_classification_analysis(data)
  
  with tab2:
      show_regression_analysis(data)
  
  with tab3:
      show_clustering_analysis(data)
  
  with tab4:
      show_model_evaluation(data)

def show_classification_analysis(data: pd.DataFrame):
  """显示分类分析"""
  st.markdown('<h2 class="sub-header">分类算法</h2>', unsafe_allow_html=True)
  
  # 选择目标变量和特征变量
  all_cols = data.columns.tolist()
  
  target_var = st.selectbox("选择目标变量（分类标签）", all_cols)
  
  if target_var:
      feature_vars = st.multiselect(
          "选择特征变量",
          [col for col in all_cols if col != target_var],
          default=[col for col in all_cols if col != target_var][:5]
      )
      
      if not feature_vars:
          st.warning("请至少选择一个特征变量")
          return
      
      # 检查目标变量的类型
      target_unique = data[target_var].nunique()
      
      if target_unique > 10:
          st.warning("目标变量的唯一值过多，可能不适合分类任务")
          return
      
      # 选择算法
      algorithm = st.selectbox(
          "选择分类算法",
          ["逻辑回归", "随机森林", "支持向量机", "朴素贝叶斯", "K近邻"]
      )
      
      # 数据预处理选项
      st.markdown("### 数据预处理")
      
      col1, col2 = st.columns(2)
      
      with col1:
          handle_missing = st.selectbox(
              "缺失值处理",
              ["删除缺失值", "均值填充", "中位数填充"]
          )
      
      with col2:
          scale_features = st.checkbox("特征标准化", value=True)
      
      test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
      
      if st.button("训练模型"):
          try:
              # 数据准备
              model_data = data[[target_var] + feature_vars].copy()
              
              # 处理缺失值
              if handle_missing == "删除缺失值":
                  model_data = model_data.dropna()
              elif handle_missing == "均值填充":
                  numeric_cols = model_data.select_dtypes(include=[np.number]).columns
                  model_data[numeric_cols] = model_data[numeric_cols].fillna(model_data[numeric_cols].mean())
              elif handle_missing == "中位数填充":
                  numeric_cols = model_data.select_dtypes(include=[np.number]).columns
                  model_data[numeric_cols] = model_data[numeric_cols].fillna(model_data[numeric_cols].median())
              
              # 编码分类变量
              le_dict = {}
              for col in model_data.columns:
                  if model_data[col].dtype == 'object':
                      le = LabelEncoder()
                      model_data[col] = le.fit_transform(model_data[col].astype(str))
                      le_dict[col] = le
              
              # 分离特征和目标
              X = model_data[feature_vars]
              y = model_data[target_var]
              
              # 划分训练测试集
              X_train, X_test, y_train, y_test = train_test_split(
                  X, y, test_size=test_size, random_state=42, stratify=y
              )
              
              # 特征标准化
              if scale_features:
                  scaler = StandardScaler()
                  X_train_scaled = scaler.fit_transform(X_train)
                  X_test_scaled = scaler.transform(X_test)
              else:
                  X_train_scaled = X_train.values
                  X_test_scaled = X_test.values
              
              # 选择和训练模型
              if algorithm == "逻辑回归":
                  model = LogisticRegression(random_state=42, max_iter=1000)
              elif algorithm == "随机森林":
                  model = RandomForestClassifier(random_state=42, n_estimators=100)
              elif algorithm == "支持向量机":
                  model = SVC(random_state=42, probability=True)
              elif algorithm == "朴素贝叶斯":
                  model = GaussianNB()
              elif algorithm == "K近邻":
                  model = KNeighborsClassifier(n_neighbors=5)
              
              # 训练模型
              model.fit(X_train_scaled, y_train)
              
              # 预测
              y_pred = model.predict(X_test_scaled)
              y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
              
              # 显示结果
              show_classification_results(y_test, y_pred, y_pred_proba, model, feature_vars, algorithm)
              
              # 保存模型到session state
              st.session_state.trained_model = {
                  'model': model,
                  'scaler': scaler if scale_features else None,
                  'feature_vars': feature_vars,
                  'target_var': target_var,
                  'le_dict': le_dict,
                  'algorithm': algorithm
              }
              
          except Exception as e:
              st.error(f"模型训练失败: {str(e)}")

def show_classification_results(y_test, y_pred, y_pred_proba, model, feature_vars, algorithm):
  """显示分类结果"""
  st.markdown("### 🎯 模型性能评估")
  
  # 计算评估指标
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')
  f1 = f1_score(y_test, y_pred, average='weighted')
  
  # 显示指标
  col1, col2, col3, col4 = st.columns(4)
  
  with col1:
      st.metric("准确率", f"{accuracy:.3f}")
  with col2:
      st.metric("精确率", f"{precision:.3f}")
  with col3:
      st.metric("召回率", f"{recall:.3f}")
  with col4:
      st.metric("F1分数", f"{f1:.3f}")
  
  # 混淆矩阵
  cm = confusion_matrix(y_test, y_pred)
  
  col1, col2 = st.columns(2)
  
  with col1:
      st.markdown("#### 混淆矩阵")
      fig_cm = px.imshow(
          cm,
          text_auto=True,
          aspect="auto",
          title="混淆矩阵",
          labels=dict(x="预测标签", y="真实标签")
      )
      st.plotly_chart(fig_cm, use_container_width=True)
  
  with col2:
      # 分类报告
      st.markdown("#### 分类报告")
      report = classification_report(y_test, y_pred, output_dict=True)
      report_df = pd.DataFrame(report).transpose()
      st.dataframe(report_df.round(3), use_container_width=True)
  
  # ROC曲线（二分类）
  if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
      st.markdown("#### ROC曲线")
      
      fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
      auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
      
      fig_roc = px.line(
          x=fpr, y=tpr,
          title=f"ROC曲线 (AUC = {auc_score:.3f})",
          labels={'x': '假正率', 'y': '真正率'}
      )
      fig_roc.add_shape(
          type="line", line=dict(dash="dash"),
          x0=0, x1=1, y0=0, y1=1
      )
      st.plotly_chart(fig_roc, use_container_width=True)
  
  # 特征重要性（如果模型支持）
  if hasattr(model, 'feature_importances_'):
      st.markdown("#### 特征重要性")
      
      importance_df = pd.DataFrame({
          '特征': feature_vars,
          '重要性': model.feature_importances_
      }).sort_values('重要性', ascending=False)
      
      fig_importance = px.bar(
          importance_df,
          x='重要性',
          y='特征',
          orientation='h',
          title="特征重要性排序"
      )
      st.plotly_chart(fig_importance, use_container_width=True)

def show_regression_analysis(data: pd.DataFrame):
  """显示回归分析"""
  st.markdown('<h2 class="sub-header">回归算法</h2>', unsafe_allow_html=True)
  
  # 选择目标变量和特征变量
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if len(numeric_cols) < 2:
      st.warning("需要至少2个数值变量进行回归分析")
      return
  
  target_var = st.selectbox("选择目标变量（连续值）", numeric_cols)
  
  if target_var:
      feature_vars = st.multiselect(
          "选择特征变量",
          [col for col in numeric_cols if col != target_var],
          default=[col for col in numeric_cols if col != target_var][:3]
      )
      
      if not feature_vars:
          st.warning("请至少选择一个特征变量")
          return
      
      # 选择算法
      algorithm = st.selectbox(
          "选择回归算法",
          ["线性回归", "岭回归", "Lasso回归", "随机森林回归", "支持向量回归"]
      )
      
      # 数据预处理选项
      st.markdown("### 数据预处理")
      
      col1, col2 = st.columns(2)
      
      with col1:
          handle_missing = st.selectbox(
              "缺失值处理",
              ["删除缺失值", "均值填充", "中位数填充"],
              key="reg_missing"
          )
      
      with col2:
          scale_features = st.checkbox("特征标准化", value=True, key="reg_scale")
      
      test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05, key="reg_test_size")
      
      if st.button("训练回归模型"):
          try:
              # 数据准备
              model_data = data[[target_var] + feature_vars].copy()
              
              # 处理缺失值
              if handle_missing == "删除缺失值":
                  model_data = model_data.dropna()
              elif handle_missing == "均值填充":
                  model_data = model_data.fillna(model_data.mean())
              elif handle_missing == "中位数填充":
                  model_data = model_data.fillna(model_data.median())
              
              # 分离特征和目标
              X = model_data[feature_vars]
              y = model_data[target_var]
              
              # 划分训练测试集
              X_train, X_test, y_train, y_test = train_test_split(
                  X, y, test_size=test_size, random_state=42
              )
              
              # 特征标准化
              if scale_features:
                  scaler = StandardScaler()
                  X_train_scaled = scaler.fit_transform(X_train)
                  X_test_scaled = scaler.transform(X_test)
              else:
                  X_train_scaled = X_train.values
                  X_test_scaled = X_test.values
              
              # 选择和训练模型
              if algorithm == "线性回归":
                  model = LinearRegression()
              elif algorithm == "岭回归":
                  model = Ridge(alpha=1.0, random_state=42)
              elif algorithm == "Lasso回归":
                  model = Lasso(alpha=1.0, random_state=42)
              elif algorithm == "随机森林回归":
                  model = RandomForestRegressor(random_state=42, n_estimators=100)
              elif algorithm == "支持向量回归":
                  model = SVR(kernel='rbf')
              
              # 训练模型
              model.fit(X_train_scaled, y_train)
              
              # 预测
              y_pred_train = model.predict(X_train_scaled)
              y_pred_test = model.predict(X_test_scaled)
              
              # 显示结果
              show_regression_results(
                  y_train, y_pred_train, y_test, y_pred_test, 
                  model, feature_vars, algorithm
              )
              
          except Exception as e:
              st.error(f"回归模型训练失败: {str(e)}")

def show_regression_results(y_train, y_pred_train, y_test, y_pred_test, model, feature_vars, algorithm):
  """显示回归结果"""
  st.markdown("### 📊 回归模型性能评估")
  
  # 计算评估指标
  train_r2 = r2_score(y_train, y_pred_train)
  test_r2 = r2_score(y_test, y_pred_test)
  train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
  test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
  train_mae = np.mean(np.abs(y_train - y_pred_train))
  test_mae = np.mean(np.abs(y_test - y_pred_test))
  
  # 显示指标
  col1, col2 = st.columns(2)
  
  with col1:
      st.markdown("#### 训练集性能")
      st.metric("R²", f"{train_r2:.4f}")
      st.metric("RMSE", f"{train_rmse:.4f}")
      st.metric("MAE", f"{train_mae:.4f}")
  
  with col2:
      st.markdown("#### 测试集性能")
      st.metric("R²", f"{test_r2:.4f}")
      st.metric("RMSE", f"{test_rmse:.4f}")
      st.metric("MAE", f"{test_mae:.4f}")
  
  # 预测vs实际值图
  col1, col2 = st.columns(2)
  
  with col1:
      # 训练集
      fig_train = px.scatter(
          x=y_train, y=y_pred_train,
          title="训练集：预测值 vs 实际值",
          labels={'x': '实际值', 'y': '预测值'}
      )
      # 添加理想线
      min_val, max_val = min(y_train.min(), y_pred_train.min()), max(y_train.max(), y_pred_train.max())
      fig_train.add_shape(
          type="line", line=dict(dash="dash", color="red"),
          x0=min_val, x1=max_val, y0=min_val, y1=max_val
      )
      st.plotly_chart(fig_train, use_container_width=True)
  
  with col2:
      # 测试集
      fig_test = px.scatter(
          x=y_test, y=y_pred_test,
          title="测试集：预测值 vs 实际值",
          labels={'x': '实际值', 'y': '预测值'}
      )
      # 添加理想线
      min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
      fig_test.add_shape(
          type="line", line=dict(dash="dash", color="red"),
          x0=min_val, x1=max_val, y0=min_val, y1=max_val
      )
      st.plotly_chart(fig_test, use_container_width=True)
  
  # 残差分析
  residuals_train = y_train - y_pred_train
  residuals_test = y_test - y_pred_test
  
  col1, col2 = st.columns(2)
  
  with col1:
      # 残差vs预测值
      fig_resid = px.scatter(
          x=y_pred_test, y=residuals_test,
          title="残差 vs 预测值",
          labels={'x': '预测值', 'y': '残差'}
      )
      fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
      st.plotly_chart(fig_resid, use_container_width=True)
  
  with col2:
      # 残差分布
      fig_resid_hist = px.histogram(
          x=residuals_test,
          title="残差分布",
          labels={'x': '残差', 'y': '频数'}
      )
      st.plotly_chart(fig_resid_hist, use_container_width=True)
  
  # 特征重要性（如果模型支持）
  if hasattr(model, 'feature_importances_'):
      st.markdown("#### 特征重要性")
      
      importance_df = pd.DataFrame({
          '特征': feature_vars,
          '重要性': model.feature_importances_
      }).sort_values('重要性', ascending=False)
      
      fig_importance = px.bar(
          importance_df,
          x='重要性',
          y='特征',
          orientation='h',
          title="特征重要性排序"
      )
      st.plotly_chart(fig_importance, use_container_width=True)
  
  elif hasattr(model, 'coef_'):
      st.markdown("#### 回归系数")
      
      coef_df = pd.DataFrame({
          '特征': feature_vars,
          '系数': model.coef_
      })
      
      if hasattr(model, 'intercept_'):
          intercept_row = pd.DataFrame({'特征': ['截距'], '系数': [model.intercept_]})
          coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
      
      st.dataframe(coef_df.round(4), use_container_width=True)

def show_clustering_analysis(data: pd.DataFrame):
  """显示聚类分析"""
  st.markdown('<h2 class="sub-header">聚类分析</h2>', unsafe_allow_html=True)
  
  numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
  
  if len(numeric_cols) < 2:
      st.warning("需要至少2个数值变量进行聚类分析")
      return
  
  # 选择特征变量
  feature_vars = st.multiselect(
      "选择用于聚类的特征变量",
      numeric_cols,
      default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
  )
  
  if len(feature_vars) < 2:
      st.warning("请至少选择2个特征变量")
      return
  
  # 选择聚类算法
  algorithm = st.selectbox(
      "选择聚类算法",
      ["K-means", "层次聚类", "DBSCAN"]
  )
  
  # 数据预处理
  st.markdown("### 数据预处理")
  
  col1, col2 = st.columns(2)
  
  with col1:
      handle_missing = st.selectbox(
          "缺失值处理",
          ["删除缺失值", "均值填充"],
          key="cluster_missing"
      )
  
  with col2:
      scale_features = st.checkbox("特征标准化", value=True, key="cluster_scale")
  
  # 算法参数
  if algorithm == "K-means":
      n_clusters = st.slider("聚类数量", 2, 10, 3)
  elif algorithm == "层次聚类":
      n_clusters = st.slider("聚类数量", 2, 10, 3)
      linkage = st.selectbox("链接方法", ["ward", "complete", "average", "single"])
  elif algorithm == "DBSCAN":
      eps = st.slider("邻域半径(eps)", 0.1, 2.0, 0.5, 0.1)
      min_samples = st.slider("最小样本数", 2, 20, 5)
  
  if st.button("执行聚类分析"):
      try:
          # 数据准备
          cluster_data = data[feature_vars].copy()
          
          # 处理缺失值
          if handle_missing == "删除缺失值":
              cluster_data = cluster_data.dropna()
          elif handle_missing == "均值填充":
              cluster_data = cluster_data.fillna(cluster_data.mean())
          
          # 特征标准化
          if scale_features:
              scaler = StandardScaler()
              X_scaled = scaler.fit_transform(cluster_data)
          else:
              X_scaled = cluster_data.values
          
          # 选择和执行聚类算法
          if algorithm == "K-means":
              clusterer = KMeans(n_clusters=n_clusters, random_state=42)
              cluster_labels = clusterer.fit_predict(X_scaled)
          elif algorithm == "层次聚类":
              clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
              cluster_labels = clusterer.fit_predict(X_scaled)
          elif algorithm == "DBSCAN":
              clusterer = DBSCAN(eps=eps, min_samples=min_samples)
              cluster_labels = clusterer.fit_predict(X_scaled)
          
          # 显示聚类结果
          show_clustering_results(cluster_data, X_scaled, cluster_labels, feature_vars, algorithm)
          
      except Exception as e:
          st.error(f"聚类分析失败: {str(e)}")

def show_clustering_results(original_data, scaled_data, cluster_labels, feature_vars, algorithm):
  """显示聚类结果"""
  st.markdown("### 🎯 聚类分析结果")
  
  # 基本统计
  n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))  # 排除噪声点(-1)
  n_noise = np.sum(cluster_labels == -1) if -1 in cluster_labels else 0
  
  col1, col2, col3 = st.columns(3)
  
  with col1:
      st.metric("聚类数量", n_clusters)
  with col2:
      st.metric("样本总数", len(cluster_labels))
  with col3:
      if n_noise > 0:
          st.metric("噪声点数量", n_noise)
      else:
          st.metric("噪声点数量", 0)
  
  # 轮廓系数
  if n_clusters > 1 and len(np.unique(cluster_labels)) > 1:
      silhouette_avg = silhouette_score(scaled_data, cluster_labels)
      st.metric("平均轮廓系数", f"{silhouette_avg:.4f}")
      
      if silhouette_avg > 0.7:
          st.success("✅ 聚类效果很好")
      elif silhouette_avg > 0.5:
          st.info("ℹ️ 聚类效果较好")
      elif silhouette_avg > 0.25:
          st.warning("⚠️ 聚类效果一般")
      else:
          st.error("❌ 聚类效果较差")
  
  # 聚类可视化
  if len(feature_vars) >= 2:
      # 添加聚类标签到原始数据
      plot_data = original_data.copy()
      plot_data['聚类'] = cluster_labels
      
      # 2D散点图
      fig_2d = px.scatter(
          plot_data,
          x=feature_vars[0],
          y=feature_vars[1],
          color='聚类',
          title=f"聚类结果 ({feature_vars[0]} vs {feature_vars[1]})",
          color_continuous_scale='viridis'
      )
      st.plotly_chart(fig_2d, use_container_width=True)
      
      # 3D散点图（如果有3个或更多特征）
      if len(feature_vars) >= 3:
          fig_3d = px.scatter_3d(
              plot_data,
              x=feature_vars[0],
              y=feature_vars[1],
              z=feature_vars[2],
              color='聚类',
              title=f"3D聚类结果",
              color_continuous_scale='viridis'
          )
          st.plotly_chart(fig_3d, use_container_width=True)
  
  # 聚类中心统计（K-means）
  if algorithm == "K-means":
      st.markdown("### 聚类中心特征")
      
      cluster_centers = []
      for i in range(n_clusters):
          cluster_mask = cluster_labels == i
          cluster_center = original_data[cluster_mask].mean()
          cluster_centers.append(cluster_center)
      
      centers_df = pd.DataFrame(cluster_centers, 
                              index=[f'聚类{i}' for i in range(n_clusters)],
                              columns=feature_vars)
      st.dataframe(centers_df.round(4), use_container_width=True)
  
  # 各聚类的样本数量
  st.markdown("### 聚类分布")
  
  cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
  
  fig_dist = px.bar(
      x=cluster_counts.index,
      y=cluster_counts.values,
      title="各聚类样本数量",
      labels={'x': '聚类标签', 'y': '样本数量'}
  )
  st.plotly_chart(fig_dist, use_container_width=True)

def show_model_evaluation(data: pd.DataFrame):
  """显示模型评估"""
  st.markdown('<h2 class="sub-header">模型评估与比较</h2>', unsafe_allow_html=True)
  
  # 检查是否有已训练的模型
  if 'trained_model' not in st.session_state:
      st.warning("⚠️ 请先训练一个模型")
      return
  
  model_info = st.session_state.trained_model
  
  st.markdown("### 📋 已训练模型信息")
  
  col1, col2, col3 = st.columns(3)
  
  with col1:
      st.info(f"**算法**: {model_info['algorithm']}")
  with col2:
      st.info(f"**目标变量**: {model_info['target_var']}")
  with col3:
      st.info(f"**特征数量**: {len(model_info['feature_vars'])}")
  
  # 交叉验证
  st.markdown("### 🔄 交叉验证")
  
  cv_folds = st.slider("交叉验证折数", 3, 10, 5)
  
  if st.button("执行交叉验证"):
      try:
          # 准备数据
          feature_vars = model_info['feature_vars']
          target_var = model_info['target_var']
          
          model_data = data[[target_var] + feature_vars].dropna()
          
          # 编码处理（如果需要）
          if model_info['le_dict']:
              for col, le in model_info['le_dict'].items():
                  if col in model_data.columns:
                      model_data[col] = le.transform(model_data[col].astype(str))
          
          X = model_data[feature_vars]
          y = model_data[target_var]
          
          # 标准化（如果需要）
          if model_info['scaler']:
              X_scaled = model_info['scaler'].fit_transform(X)
          else:
              X_scaled = X.values
          
          # 执行交叉验证
          model = model_info['model']
          
          if hasattr(model, 'predict_proba'):  # 分类模型
              cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
              metric_name = "准确率"
          else:  # 回归模型
              cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
              metric_name = "R²"
          
          # 显示交叉验证结果
          col1, col2, col3 = st.columns(3)
          
          with col1:
              st.metric(f"平均{metric_name}", f"{cv_scores.mean():.4f}")
          with col2:
              st.metric("标准差", f"{cv_scores.std():.4f}")
          with col3:
              st.metric("95%置信区间", f"±{1.96 * cv_scores.std():.4f}")
          
          # 交叉验证分数分布
          fig_cv = px.box(
              y=cv_scores,
              title=f"交叉验证{metric_name}分布"
          )
          st.plotly_chart(fig_cv, use_container_width=True)
          
      except Exception as e:
          st.error(f"交叉验证失败: {str(e)}")
  
  # 模型预测
  st.markdown("### 🔮 模型预测")
  
  st.write("输入特征值进行预测：")
  
  feature_vars = model_info['feature_vars']
  input_values = {}
  
  cols = st.columns(min(3, len(feature_vars)))
  
  for i, feature in enumerate(feature_vars):
      with cols[i % 3]:
          # 获取该特征的统计信息
          feature_data = data[feature].dropna()
          min_val = float(feature_data.min())
          max_val = float(feature_data.max())
          mean_val = float(feature_data.mean())
          
          input_values[feature] = st.number_input(
              f"{feature}",
              min_value=min_val,
              max_value=max_val,
              value=mean_val,
              key=f"predict_{feature}"
          )
  
  if st.button("进行预测"):
      try:
          # 准备输入数据
          input_df = pd.DataFrame([input_values])
          
          # 标准化（如果需要）
          if model_info['scaler']:
              input_scaled = model_info['scaler'].transform(input_df)
          else:
              input_scaled = input_df.values
          
          # 预测
          model = model_info['model']
          prediction = model.predict(input_scaled)[0]
          
          st.success(f"🎯 预测结果: {prediction:.4f}")
          
          # 如果是分类模型，显示概率
          if hasattr(model, 'predict_proba'):
              probabilities = model.predict_proba(input_scaled)[0]
              
              st.markdown("#### 预测概率")
              
              prob_df = pd.DataFrame({
                  '类别': model.classes_,
                  '概率': probabilities
              })
              
              fig_prob = px.bar(
                  prob_df,
                  x='类别',
                  y='概率',
                  title="各类别预测概率"
              )
              st.plotly_chart(fig_prob, use_container_width=True)
          
      except Exception as e:
          st.error(f"预测失败: {str(e)}")
