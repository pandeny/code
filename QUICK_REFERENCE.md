# 负荷变化可解释性模型 - 快速参考

## 快速开始

### 1. 运行演示（推荐首次使用）
```bash
python load_interpretability_demo.py
```
这将生成：
- `/tmp/load_interpretability_demo.png` - 可视化结果
- `/tmp/load_interpretability_report.txt` - 分析报告

### 2. 在预测系统中使用
```bash
python train_household_forecast.py
```
选择"预测模式"，输入日期后自动生成可解释性分析。

## 主要功能速查

### 阶段划分
```python
states, state_means, segments = hmm_load_segmentation(
    load_values, 
    n_states='auto',      # 自动选择状态数
    min_states=3,         # 最小3个状态
    max_states=5,         # 最大5个状态
    min_segment_length=8  # 最小段长度8个点
)
```

### 生成解释
```python
explanations = explain_load_changes(
    segments,      # 阶段信息
    feat_df,       # 特征数据框
    pred_times,    # 时间点列表
    load_values    # 负荷值数组
)
```

### 生成报告
```python
generate_explanation_report(explanations, 'report.txt')
```

### 可视化
```python
visualize_explanations(explanations, 'viz.png')
```

## 输出文件说明

### 预测模式输出
| 文件名 | 说明 |
|--------|------|
| `prediction_with_stages_YYYYMMDD.png` | 预测结果与阶段划分可视化 |
| `explanation_report_YYYYMMDD.txt` | 文本格式的解释报告 |
| `explanation_viz_YYYYMMDD.png` | 可解释性分析可视化 |
| `explanation_YYYYMMDD.json` | JSON格式的分析数据 |
| `prediction_detail_YYYYMMDD.csv` | 详细预测数据 |
| `prediction_summary_YYYYMMDD.csv` | 预测汇总 |

## 解释维度

### 1. 阶段分析
- 负荷水平（低/中低/中高/高）
- 时间段特征
- 环境因素影响
- 关键驱动因素

### 2. 趋势分析
- 变化量和变化率
- 变化趋势分类
- 变化原因解释

### 3. 特征重要性
- 相关性系数
- 重要特征排序
- 影响方向

### 4. 环境影响
- 统计特征（均值、标准差、范围）
- 波动幅度评估

## 常见问题

### Q: 如何调整阶段数量？
A: 修改`hmm_load_segmentation`的`min_states`和`max_states`参数。

### Q: 如何添加新的环境特征？
A: 在数据源中添加特征列，系统会自动识别和分析。

### Q: 报告语言是否可以切换？
A: 代码中使用中文，可通过修改字符串模板切换语言。

### Q: 如何优化分段效果？
A: 调整`min_segment_length`参数，增大值可减少分段数量。

## 性能提示

1. **数据量大时**：可以先对数据进行降采样
2. **HMM训练慢**：设置更小的`max_states`值
3. **内存占用高**：分批处理长时间序列

## 示例代码

### 完整分析流程
```python
import pandas as pd
import numpy as np

# 1. 加载数据
df = pd.read_csv('load_data.csv')
df = df.set_index('time')

# 2. 阶段划分
states, state_means, segments = hmm_load_segmentation(
    df['load'].values
)

# 3. 生成解释
explanations = explain_load_changes(
    segments, 
    df, 
    df.index.tolist(), 
    df['load'].values
)

# 4. 生成报告
generate_explanation_report(explanations, 'report.txt')

# 5. 可视化
visualize_explanations(explanations, 'viz.png')

# 6. 打印摘要
for seg in explanations['segment_analysis']:
    print(f"阶段{seg['segment_id']}: {seg['load_level']}")
    print(f"  因素: {seg['key_factors'][0]}")
```

## 扩展建议

### 添加新的解释维度
在`explain_load_changes()`中添加新的分析逻辑：
```python
# 示例：添加负荷波动分析
explanations['volatility_analysis'] = {
    'daily_volatility': np.std(load_values),
    'peak_to_avg_ratio': np.max(load_values) / np.mean(load_values)
}
```

### 自定义可视化
基于`matplotlib`创建自定义图表：
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
# 添加自定义绘图逻辑
ax.plot(...)
plt.savefig('custom_viz.png')
```

## 更多信息

- 详细文档：`INTERPRETABILITY_MODEL.md`
- 实现总结：`IMPLEMENTATION_SUMMARY.md`
- 项目说明：`README.md`

## 技术支持

遇到问题？
1. 查看详细文档
2. 运行演示脚本验证环境
3. 提交GitHub Issue

---
最后更新：2024年10月
