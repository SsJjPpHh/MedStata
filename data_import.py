import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import pyreadstat
from utils.data_validation import DataValidator
from utils.file_handlers import FileHandler

def show():
  """显示数据导入页面"""
  st.markdown('<h1 class="main-header">📊 数据导入</h1>', unsafe_allow_html=True)
  
  # 创建标签页
  tab1, tab2, tab3 = st.tabs(["文件上传", "数据预览", "数据验证"])
  
  with tab1:
      show_file_upload()
  
  with tab2:
      show_data_preview()
  
  with tab3:
      show_data_validation()

def show_file_upload():
  """文件上传界面"""
  st.markdown('<h2 class="sub-header">上传数据文件</h2>', unsafe_allow_html=True)
  
  # 文件上传组件
  uploaded_file = st.file_uploader(
      "选择数据文件",
      type=['csv', 'xlsx', 'xls', 'json', 'sav'],
      help="支持CSV、Excel、JSON和SPSS文件格式"
  )
  
  if uploaded_file is not None:
      try:
          # 根据文件类型读取数据
          file_handler = FileHandler()
          data = file_handler.load_file(uploaded_file)
          
          if data is not None:
              st.session_state.data = data
              st.session_state.current_dataset = uploaded_file.name
              
              st.success(f"✅ 成功加载文件: {uploaded_file.name}")
              st.info(f"数据维度: {data.shape[0]} 行 × {data.shape[1]} 列")
              
      except Exception as e:
          st.error(f"❌ 文件加载失败: {str(e)}")
  
  # 示例数据选项
  st.markdown("---")
  st.markdown("### 或使用示例数据")
  
  col1, col2, col3 = st.columns(3)
  
  with col1:
      if st.button("心脏病数据集"):
          data = load_sample_heart_data()
          st.session_state.data = data
          st.session_state.current_dataset = "heart_disease_sample"
          st.success("✅ 已加载心脏病示例数据")
  
  with col2:
      if st.button("癌症生存数据"):
          data = load_sample_survival_data()
          st.session_state.data = data
          st.session_state.current_dataset = "cancer_survival_sample"
          st.success("✅ 已加载癌症生存示例数据")
  
  with col3:
      if st.button("临床试验数据"):
          data = load_sample_clinical_data()
          st.session_state.data = data
          st.session_state.current_dataset = "clinical_trial_sample"
          st.success("✅ 已加载临床试验示例数据")

def show_data_preview():
  """数据预览界面"""
  if st.session_state.data is None:
      st.warning("⚠️ 请先上传数据文件")
      return
  
  data = st.session_state.data
  
  st.markdown('<h2 class="sub-header">数据预览</h2>', unsafe_allow_html=True)
  
  # 数据基本信息
  col1, col2, col3, col4 = st.columns(4)
  
  with col1:
      st.metric("总行数", data.shape[0])
  with col2:
      st.metric("总列数", data.shape[1])
  with col3:
      st.metric("缺失值", data.isnull().sum().sum())
  with col4:
      st.metric("数值列", len(data.select_dtypes(include=[np.number]).columns))
  
  # 数据表格显示
  st.markdown("### 数据表格")
  
  # 显示选项
  col1, col2 = st.columns(2)
  with col1:
      show_rows = st.slider("显示行数", 5, min(100, len(data)), 10)
  with col2:
      show_all_cols = st.checkbox("显示所有列", value=False)
  
  if show_all_cols:
      st.dataframe(data.head(show_rows), use_container_width=True)
  else:
      # 只显示前几列
      max_cols = min(10, len(data.columns))
      st.dataframe(data.iloc[:show_rows, :max_cols], use_container_width=True)
  
  # 数据类型信息
  st.markdown("### 数据类型")
  dtype_df = pd.DataFrame({
      '列名': data.columns,
      '数据类型': data.dtypes.astype(str),
      '非空值数量': data.count(),
      '缺失值数量': data.isnull().sum(),
      '缺失率(%)': (data.isnull().sum() / len(data) * 100).round(2)
  })
  st.dataframe(dtype_df, use_container_width=True)

def show_data_validation():
  """数据验证界面"""
  if st.session_state.data is None:
      st.warning("⚠️ 请先上传数据文件")
      return
  
  data = st.session_state.data
  validator = DataValidator()
  
  st.markdown('<h2 class="sub-header">数据质量检查</h2>', unsafe_allow_html=True)
  
  # 执行数据验证
  if st.button("🔍 开始数据验证"):
      with st.spinner("正在验证数据质量..."):
          validation_results = validator.validate_dataset(data)
          
          # 显示验证结果
          if validation_results['is_valid']:
              st.success("✅ 数据验证通过！")
          else:
              st.error("❌ 数据存在质量问题")
          
          # 详细验证报告
          st.markdown("### 验证报告")
          
          for category, issues in validation_results['issues'].items():
              if issues:
                  st.markdown(f"**{category}:**")
                  for issue in issues:
                      st.warning(f"⚠️ {issue}")
          
          # 数据质量评分
          quality_score = validation_results.get('quality_score', 0)
          st.markdown("### 数据质量评分")
          
          col1, col2 = st.columns([1, 3])
          with col1:
              st.metric("质量评分", f"{quality_score}/100")
          with col2:
              if quality_score >= 90:
                  st.success("🌟 优秀 - 数据质量很高")
              elif quality_score >= 70:
                  st.info("👍 良好 - 数据质量较好")
              elif quality_score >= 50:
                  st.warning("⚠️ 一般 - 需要改进数据质量")
              else:
                  st.error("❌ 差 - 数据质量需要大幅改进")

def load_sample_heart_data():
  """加载心脏病示例数据"""
  np.random.seed(42)
  n_samples = 300
  
  data = pd.DataFrame({
      'age': np.random.normal(55, 10, n_samples).astype(int),
      'sex': np.random.choice([0, 1], n_samples),
      'chest_pain': np.random.choice([0, 1, 2, 3], n_samples),
      'resting_bp': np.random.normal(130, 20, n_samples).astype(int),
      'cholesterol': np.random.normal(240, 50, n_samples).astype(int),
      'fasting_bs': np.random.choice([0, 1], n_samples),
      'max_hr': np.random.normal(150, 25, n_samples).astype(int),
      'exercise_angina': np.random.choice([0, 1], n_samples),
      'heart_disease': np.random.choice([0, 1], n_samples)
  })
  
  return data

def load_sample_survival_data():
  """加载生存分析示例数据"""
  np.random.seed(42)
  n_samples = 200
  
  data = pd.DataFrame({
      'patient_id': range(1, n_samples + 1),
      'age': np.random.normal(65, 12, n_samples).astype(int),
      'gender': np.random.choice(['M', 'F'], n_samples),
      'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_samples),
      'treatment': np.random.choice(['A', 'B', 'C'], n_samples),
      'survival_time': np.random.exponential(24, n_samples).round(1),
      'event': np.random.choice([0, 1], n_samples),
      'tumor_size': np.random.normal(3.5, 1.5, n_samples).round(1)
  })
  
  return data

def load_sample_clinical_data():
  """加载临床试验示例数据"""
  np.random.seed(42)
  n_samples = 150
  
  data = pd.DataFrame({
      'subject_id': range(1, n_samples + 1),
      'group': np.random.choice(['Treatment', 'Control'], n_samples),
      'baseline_score': np.random.normal(50, 10, n_samples).round(1),
      'week_4_score': np.random.normal(45, 12, n_samples).round(1),
      'week_8_score': np.random.normal(40, 15, n_samples).round(1),
      'week_12_score': np.random.normal(35, 18, n_samples).round(1),
      'adverse_events': np.random.choice([0, 1], n_samples),
      'dropout': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
  })
  
  return data
