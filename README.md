# code

## 项目概述

家庭负荷预测与可解释性分析系统。

### 主要功能

1. **负荷预测** - 基于LSTM的家庭负荷预测模型
2. **智能阶段划分** - 使用HMM自动识别负荷变化阶段
3. **可解释性分析** - 解释负荷变化的原因和影响因素

### 快速开始

#### 1. 运行负荷预测系统
```bash
python train_household_forecast.py
```

#### 2. 运行可解释性模型演示
```bash
python load_interpretability_demo.py
```

### 文档

- [负荷变化可解释性模型详细文档](INTERPRETABILITY_MODEL.md)

### 特性

- ✅ LSTM负荷预测模型
- ✅ 智能负荷阶段划分（HMM）
- ✅ 负荷变化原因解释
- ✅ 环境因素影响分析
- ✅ 特征重要性评估
- ✅ 可视化分析报告

### 依赖

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- TensorFlow/Keras
- Scikit-learn
- SciPy
