# MedStats-Pro 优化项目结构

```
MedStats-Pro/
├── 📁 config/                      # 配置文件
│   ├── settings.py                 # 应用配置
│   ├── database.py                 # 数据库配置
│   └── logging_config.py           # 日志配置
│
├── 📁 app/                         # 主应用目录
│   ├── 📁 pages/                   # Streamlit 页面
│   │   ├── 01_🏠_Home.py           # 主页
│   │   ├── 02_📊_Data_Import.py    # 数据导入
│   │   ├── 03_📈_Descriptive_Stats.py  # 描述性统计
│   │   ├── 04_🔬_Advanced_Stats.py     # 高级统计
│   │   ├── 05_🤖_Machine_Learning.py   # 机器学习
│   │   ├── 06_📉_Medical_Plots.py      # 医学图表
│   │   ├── 07_⏱️_Survival_Analysis.py  # 生存分析
│   │   ├── 08_🔄_Meta_Analysis.py      # Meta分析
│   │   └── 09_📋_Report_Generator.py   # 报告生成
│   │
│   ├── 📁 core/                    # 核心功能模块
│   │   ├── __init__.py
│   │   ├── statistical_methods.py  # 统计方法
│   │   ├── ml_algorithms.py        # 机器学习算法
│   │   ├── plot_generators.py      # 图表生成
│   │   ├── data_processor.py       # 数据处理
│   │   ├── causal_inference.py     # 因果推断
│   │   └── survival_methods.py     # 生存分析方法
│   │
│   ├── 📁 utils/                   # 工具函数
│   │   ├── __init__.py
│   │   ├── data_validation.py      # 数据验证
│   │   ├── file_handlers.py        # 文件处理
│   │   ├── calculation_helpers.py  # 计算辅助
│   │   ├── visualization_utils.py  # 可视化工具
│   │   ├── streamlit_components.py # Streamlit组件
│   │   └── export_utils.py         # 导出工具
│   │
│   ├── 📁 templates/               # 报告模板
│   │   ├── statistical_report.html
│   │   ├── ml_report.html
│   │   ├── survival_report.html
│   │   └── comprehensive_report.html
│   │
│   ├── 📁 assets/                  # 静态资源
│   │   ├── css/
│   │   │   └── custom_styles.css
│   │   ├── images/
│   │   │   └── logo.png
│   │   └── data/
│   │       └── sample_datasets/
│   │
│   └── 📁 components/              # 自定义组件
│       ├── __init__.py
│       ├── sidebar.py              # 侧边栏组件
│       ├── data_uploader.py        # 数据上传组件
│       ├── result_display.py       # 结果展示组件
│       └── plot_container.py       # 图表容器组件
│
├── 📁 tests/                       # 测试文件
│   ├── __init__.py
│   ├── test_statistical_methods.py
│   ├── test_data_processor.py
│   └── test_ml_algorithms.py
│
├── 📁 docs/                        # 文档
│   ├── README.md
│   ├── user_guide.md
│   └── api_reference.md
│
├── 📁 data/                        # 数据目录
│   ├── uploads/                    # 用户上传数据
│   ├── processed/                  # 处理后数据
│   └── exports/                    # 导出文件
│
├── main.py                         # Streamlit 主入口
├── requirements.txt                # 依赖包
├── setup.py                        # 安装配置
├── .streamlit/                     # Streamlit 配置
│   └── config.toml
└── README.md                       # 项目说明
```

## 🔧 技术栈组合

### 核心框架
- **Streamlit**: Web界面框架
- **Pandas**: 数据处理
- **NumPy**: 数值计算
- **SciPy**: 科学计算

### 统计分析
- **Statsmodels**: 统计建模
- **Scikit-learn**: 机器学习
- **Lifelines**: 生存分析
- **Pingouin**: 医学统计

### 可视化
- **Plotly**: 交互式图表
- **Matplotlib/Seaborn**: 静态图表
- **Altair**: 声明式可视化

### 报告生成
- **Jinja2**: 模板引擎
- **WeasyPrint**: PDF生成
- **Openpyxl**: Excel导出

## 🎯 优化特点

1. **模块化设计**: 清晰的功能分离
2. **Streamlit多页面**: 使用官方多页面架构
3. **组件化**: 可复用的UI组件
4. **配置管理**: 统一的配置系统
5. **测试覆盖**: 完整的测试框架
6. **文档完善**: 用户指南和API文档
