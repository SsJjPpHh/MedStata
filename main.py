import streamlit as st
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.app_config import setup_page_config
from pages import (
  data_import, descriptive_stats, advanced_stats, 
  machine_learning, medical_plots, survival_analysis, 
  meta_analysis, report_generator
)

def main():
  """主程序入口"""
  # 设置页面配置
  setup_page_config()
  
  # 侧边栏导航
  st.sidebar.title("🏥 MedStats")
  st.sidebar.markdown("医学统计分析平台")
  
  # 页面选择
  pages = {
      "📊 数据导入": data_import,
      "📈 描述性统计": descriptive_stats,
      "🔬 高级统计": advanced_stats,
      "🤖 机器学习": machine_learning,
      "📉 医学图表": medical_plots,
      "⏱️ 生存分析": survival_analysis,
      "📋 荟萃分析": meta_analysis,
      "📄 报告生成": report_generator
  }
  
  selected_page = st.sidebar.selectbox(
      "选择功能模块",
      list(pages.keys())
  )
  
  # 显示选中的页面
  if selected_page in pages:
      pages[selected_page].show()
  
  # 侧边栏信息
  st.sidebar.markdown("---")
  st.sidebar.info(
      "💡 **使用提示**\n\n"
      "1. 先导入数据\n"
      "2. 选择分析方法\n"
      "3. 查看结果\n"
      "4. 生成报告"
  )

if __name__ == "__main__":
  main()
