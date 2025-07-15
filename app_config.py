import streamlit as st
import pandas as pd
from typing import Dict, Any

def setup_page_config():
  """设置页面配置"""
  st.set_page_config(
      page_title="MedStats - 医学统计分析平台",
      page_icon="🏥",
      layout="wide",
      initial_sidebar_state="expanded"
  )
  
  # 自定义CSS样式
  st.markdown("""
  <style>
  .main-header {
      font-size: 2.5rem;
      color: #1f77b4;
      text-align: center;
      margin-bottom: 2rem;
  }
  .sub-header {
      font-size: 1.5rem;
      color: #ff7f0e;
      margin: 1rem 0;
  }
  .metric-card {
      background-color: #f0f2f6;
      padding: 1rem;
      border-radius: 0.5rem;
      border-left: 4px solid #1f77b4;
  }
  .warning-box {
      background-color: #fff3cd;
      border: 1px solid #ffeaa7;
      border-radius: 0.25rem;
      padding: 0.75rem;
      margin: 1rem 0;
  }
  </style>
  """, unsafe_allow_html=True)

def initialize_session_state():
  """初始化会话状态"""
  if 'data' not in st.session_state:
      st.session_state.data = None
  if 'analysis_results' not in st.session_state:
      st.session_state.analysis_results = {}
  if 'current_dataset' not in st.session_state:
      st.session_state.current_dataset = None

class AppConfig:
  """应用配置类"""
  
  # 支持的文件格式
  SUPPORTED_FORMATS = {
      'csv': 'CSV文件',
      'xlsx': 'Excel文件',
      'json': 'JSON文件',
      'sav': 'SPSS文件'
  }
  
  # 统计方法配置
  STATISTICAL_METHODS = {
      'descriptive': {
          'name': '描述性统计',
          'methods': ['均值', '中位数', '标准差', '四分位数', '偏度', '峰度']
      },
      'inferential': {
          'name': '推断统计',
          'methods': ['t检验', '卡方检验', '方差分析', '非参数检验']
      },
      'correlation': {
          'name': '相关分析',
          'methods': ['皮尔逊相关', '斯皮尔曼相关', '偏相关']
      }
  }
  
  # 机器学习算法配置
  ML_ALGORITHMS = {
      'classification': {
          'name': '分类算法',
          'algorithms': ['逻辑回归', '随机森林', '支持向量机', 'XGBoost']
      },
      'regression': {
          'name': '回归算法',
          'algorithms': ['线性回归', '岭回归', '随机森林回归', 'XGBoost回归']
      },
      'clustering': {
          'name': '聚类算法',
          'algorithms': ['K-means', '层次聚类', 'DBSCAN']
      }
  }
  
  # 图表类型配置
  PLOT_TYPES = {
      'basic': ['直方图', '箱线图', '散点图', '折线图'],
      'medical': ['生存曲线', 'ROC曲线', '森林图', '漏斗图'],
      'advanced': ['热图', '小提琴图', '雷达图', '桑基图']
  }
  
  @staticmethod
  def get_default_colors():
      """获取默认颜色方案"""
      return [
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf'
      ]
