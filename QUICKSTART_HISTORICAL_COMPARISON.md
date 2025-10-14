# Quick Start Guide - Historical Load Comparison

## 快速开始 - 历史负荷对比分析

### Prerequisites / 前提条件

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### Run Demo / 运行演示

```bash
# Basic load interpretability demo
# 基础负荷可解释性演示
python load_interpretability_demo.py

# Historical comparison demo (NEW!)
# 历史对比分析演示 (新功能!)
python historical_comparison_demo.py

# Run tests
# 运行测试
python test_historical_comparison.py
```

### Output Files / 输出文件

**Basic Demo:**
- `/tmp/load_interpretability_demo.png` - Load segmentation visualization
- `/tmp/load_interpretability_report.txt` - Analysis report

**Historical Comparison Demo:**
- `/tmp/historical_comparison_demo.png` - Historical comparison visualization (8 subplots)
- `/tmp/historical_comparison_report.txt` - Detailed comparison report

### Key Features / 主要功能

1. **Stage Count Comparison / 阶段数量对比**
   - Analyze increase/decrease in stage numbers
   - 分析阶段数量的增加或减少

2. **Stage Alignment / 阶段对齐**
   - Match stages between current and historical data
   - 匹配当前与历史数据的阶段

3. **Difference Detection / 差异检测**
   - Identify stages with >20% load change
   - 识别负荷变化超过20%的阶段

4. **Behavior Explanations / 行为解释**
   - Explain differences based on human behavior patterns
   - 基于人类行为模式解释差异

5. **Environmental Factors / 环境因素**
   - Consider temperature, humidity, cloud cover
   - 考虑温度、湿度、云量等因素

### Example Output / 示例输出

```
Stage count: 24 (current) vs 28 (historical)
Change: -4 stages (-14.3%)
Trend: 减少 (Decreasing)

Reasons:
  1. 用电行为更加规律，负荷模式简化
  2. 家庭成员减少或外出时间增加
  3. 减少了用电设备使用或优化了用电习惯

Significant differences: 1 stage

Stage 4 (7.2h-9.0h): +0.557 kW (+27.8%)
  Explanation: 早高峰时段负荷增加，可能是：起床时间提前、
  早餐准备更复杂、或增加了热水器/咖啡机使用
```

### Documentation / 文档

- **Full Guide:** [HISTORICAL_COMPARISON_GUIDE.md](HISTORICAL_COMPARISON_GUIDE.md)
- **Model Overview:** [INTERPRETABILITY_MODEL.md](INTERPRETABILITY_MODEL.md)
- **Implementation:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

### Use in Code / 代码中使用

```python
from train_household_forecast import compare_with_historical_stages

# Prepare your data
current_segments = [...]  # Current load segments
historical_segments = [...]  # Historical load segments
current_feat_df = pd.DataFrame(...)  # Current features
historical_feat_df = pd.DataFrame(...)  # Historical features

# Run comparison
comparison = compare_with_historical_stages(
    current_segments, historical_segments,
    current_feat_df, historical_feat_df,
    current_times, historical_times,
    current_load, historical_load
)

# Access results
print(comparison['stage_count_comparison'])
print(comparison['aligned_stages'])
print(comparison['significant_differences'])
print(comparison['behavior_explanations'])
```

### Understanding the Visualization / 理解可视化结果

The historical comparison visualization contains 8 subplots:
历史对比可视化包含8个子图：

1. **Current Load Curve** - Current load with stage segmentation
   当前负荷曲线 - 显示当前负荷和阶段划分

2. **Historical Load Curve** - Historical load with stage segmentation
   历史负荷曲线 - 显示历史负荷和阶段划分

3. **Stage Count Bar Chart** - Compare number of stages
   阶段数量柱状图 - 对比阶段数量

4. **Aligned Stage Comparison** - Side-by-side load comparison
   对齐阶段对比 - 并排对比负荷水平

5. **Load Difference Percentage** - % change for each stage
   负荷差异百分比 - 每个阶段的变化百分比

6. **Significant Differences Summary** - Text description
   显著差异摘要 - 文字描述

7. **Temperature Comparison** - Current vs historical temperature
   温度对比 - 当前与历史温度对比

8. **Behavior Analysis** - Overall behavior patterns
   行为分析 - 总体行为模式分析

### Troubleshooting / 故障排除

**Problem:** ModuleNotFoundError for tensorflow
**Solution:** Use standalone demo scripts (load_interpretability_demo.py, historical_comparison_demo.py) which don't require tensorflow

**问题：** 缺少 tensorflow 模块
**解决方案：** 使用独立演示脚本（不需要 tensorflow）

**Problem:** Chinese characters not displaying
**Solution:** Install Chinese fonts or the visualization will still work with default fonts

**问题：** 中文字符显示异常
**解决方案：** 安装中文字体，或使用默认字体（功能正常）

### Support / 支持

For issues or questions:
如有问题或疑问：

- Check the [Full Documentation](HISTORICAL_COMPARISON_GUIDE.md)
  查看[完整文档](HISTORICAL_COMPARISON_GUIDE.md)
- Open a GitHub Issue
  在 GitHub 上提交 Issue
- Review the [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
  查看[实现总结](IMPLEMENTATION_SUMMARY.md)

### What's Next / 下一步

- [ ] Try with your own load data
  使用您自己的负荷数据进行测试
- [ ] Integrate into forecasting system
  集成到预测系统中
- [ ] Customize behavior explanation rules
  自定义行为解释规则
- [ ] Explore different time periods (day vs weekend, summer vs winter)
  探索不同时间段的对比（工作日vs周末，夏季vs冬季）
