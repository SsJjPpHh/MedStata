import pandas as pd
import numpy as np
import json
import pyreadstat
from typing import Optional, Dict, Any
import streamlit as st
from io import StringIO, BytesIO

class FileHandler:
  """文件处理器类"""
  
  def __init__(self):
      self.supported_formats = {
          'csv': self._read_csv,
          'xlsx': self._read_excel,
          'xls': self._read_excel,
          'json': self._read_json,
          'sav': self._read_spss
      }
  
  def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
      """
      加载上传的文件
      
      Args:
          uploaded_file: Streamlit上传的文件对象
          
      Returns:
          pandas DataFrame或None
      """
      if uploaded_file is None:
          return None
      
      # 获取文件扩展名
      file_extension = uploaded_file.name.split('.')[-1].lower()
      
      if file_extension not in self.supported_formats:
          raise ValueError(f"不支持的文件格式: {file_extension}")
      
      # 调用相应的读取函数
      return self.supported_formats[file_extension](uploaded_file)
  
  def _read_csv(self, uploaded_file) -> pd.DataFrame:
      """读取CSV文件"""
      try:
          # 尝试不同的编码
          encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1']
          
          for encoding in encodings:
              try:
                  # 重置文件指针
                  uploaded_file.seek(0)
                  content = uploaded_file.read().decode(encoding)
                  
                  # 尝试不同的分隔符
                  separators = [',', ';', '\t', '|']
                  
                  for sep in separators:
                      try:
                          df = pd.read_csv(StringIO(content), sep=sep)
                          
                          # 验证读取结果
                          if len(df.columns) > 1 and len(df) > 0:
                              return df
                      except:
                          continue
                          
              except UnicodeDecodeError:
                  continue
          
          # 如果所有尝试都失败，使用默认方式
          uploaded_file.seek(0)
          return pd.read_csv(uploaded_file)
          
      except Exception as e:
          raise Exception(f"CSV文件读取失败: {str(e)}")
  
  def _read_excel(self, uploaded_file) -> pd.DataFrame:
      """读取Excel文件"""
      try:
          # 读取Excel文件
          excel_file = pd.ExcelFile(uploaded_file)
          
          # 如果有多个工作表，让用户选择
          if len(excel_file.sheet_names) > 1:
              st.info(f"检测到多个工作表: {', '.join(excel_file.sheet_names)}")
              selected_sheet = st.selectbox(
                  "选择要导入的工作表:",
                  excel_file.sheet_names
              )
              return pd.read_excel(uploaded_file, sheet_name=selected_sheet)
          else:
              return pd.read_excel(uploaded_file)
              
      except Exception as e:
          raise Exception(f"Excel文件读取失败: {str(e)}")
  
  def _read_json(self, uploaded_file) -> pd.DataFrame:
      """读取JSON文件"""
      try:
          content = uploaded_file.read().decode('utf-8')
          data = json.loads(content)
          
          # 尝试不同的JSON结构
          if isinstance(data, list):
              return pd.DataFrame(data)
          elif isinstance(data, dict):
              # 如果是嵌套字典，尝试展平
              return pd.json_normalize(data)
          else:
              raise ValueError("不支持的JSON结构")
              
      except Exception as e:
          raise Exception(f"JSON文件读取失败: {str(e)}")
  
  def _read_spss(self, uploaded_file) -> pd.DataFrame:
      """读取SPSS文件"""
      try:
          # 将上传的文件保存到临时位置
          temp_path = f"/tmp/{uploaded_file.name}"
          with open(temp_path, "wb") as f:
              f.write(uploaded_file.read())
          
          # 使用pyreadstat读取SPSS文件
          df, meta = pyreadstat.read_sav(temp_path)
          
          # 清理临时文件
          import os
          os.remove(temp_path)
          
          return df
          
      except Exception as e:
          raise Exception(f"SPSS文件读取失败: {str(e)}")
  
  def export_data(self, data: pd.DataFrame, filename: str, format: str) -> bytes:
      """
      导出数据到指定格式
      
      Args:
          data: 要导出的数据框
          filename: 文件名
          format: 导出格式
          
      Returns:
          文件字节数据
      """
      if format.lower() == 'csv':
          return data.to_csv(index=False).encode('utf-8')
      
      elif format.lower() in ['xlsx', 'excel']:
          buffer = BytesIO()
          with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
              data.to_excel(writer, index=False, sheet_name='Data')
          return buffer.getvalue()
      
      elif format.lower() == 'json':
          return data.to_json(orient='records', indent=2).encode('utf-8')
      
      else:
          raise ValueError(f"不支持的导出格式: {format}")
  
  def get_file_info(self, uploaded_file) -> Dict[str, Any]:
      """获取文件信息"""
      return {
          'name': uploaded_file.name,
          'size': uploaded_file.size,
          'type': uploaded_file.type,
          'extension': uploaded_file.name.split('.')[-1].lower()
      }
