# 历史负荷阶段对比分析功能

## 概述

本功能扩展了负荷变化可解释性模型，新增了与历史负荷数据进行对比分析的能力。该功能可以帮助用户理解负荷模式随时间的变化，识别异常行为，并基于人的行为模式提供解释。

## 主要功能

### 1. 阶段数量变化分析
- 对比当前与历史负荷的阶段数量
- 分析阶段数增加或减少的原因
- 提供基于用电行为的解释

**输出示例：**
```
当前阶段数: 24
历史阶段数: 28
变化: -4 个阶段 (-14.3%)
趋势: 减少

原因分析:
  负荷阶段数减少4个，可能原因：
  1. 用电行为更加规律，负荷模式简化
  2. 家庭成员减少或外出时间增加
  3. 减少了用电设备使用或优化了用电习惯
```

### 2. 逐阶段对齐分析
- 基于时间重叠度匹配当前和历史阶段
- 计算每对对齐阶段的负荷差异
- 结合环境特征差异（温度、湿度、云量等）

**对齐策略：**
- 优先匹配时间重叠最大的阶段
- 如果时间重叠相同，选择中心点时间最接近的阶段
- 考虑负荷水平相似性

**输出示例：**
```
当前阶段4 ↔ 历史阶段6:
  时间范围: 7.2h-9.0h (当前) vs 7.8h-9.0h (历史)
  负荷水平: 2.5608 kW (当前) vs 2.0035 kW (历史)
  负荷差异: +0.5573 kW (+27.8%)
  环境特征差异:
    • temperature: 18.5°C (当前) vs 15.2°C (历史), 差异: +3.3°C
```

### 3. 差异显著阶段识别
- 自动识别负荷差异超过20%的阶段
- 标记为需要重点关注的阶段
- 提供详细的差异分析

### 4. 基于人类行为的解释
- 根据时间段（早高峰、上午、午间、下午、晚高峰、夜间）提供针对性解释
- 结合负荷增减趋势分析用电行为变化
- 考虑环境因素影响（温度、湿度等）

**行为解释示例：**

**早高峰时段（6-9h）负荷增加：**
- 起床时间提前
- 早餐准备更复杂
- 增加了热水器/咖啡机使用

**上午时段（9-12h）负荷增加：**
- 在家办公
- 使用更多电器
- 家庭成员未外出

**晚高峰时段（18-22h）负荷减少：**
- 回家时间推迟
- 简化晚餐准备
- 减少娱乐设备使用
- 改善照明和空调使用习惯

### 5. 总体行为模式分析
- 统计增加和减少的阶段数量
- 判断整体趋势（增加为主、减少为主、平衡）
- 提供宏观层面的行为解释

**输出示例：**
```
共识别出3个差异显著的负荷阶段
整体趋势：负荷增加为主(2个阶段增加，1个阶段减少)
可能原因：家庭活动增加、在家时间延长、新增用电设备、或季节性用电需求变化
```

## 使用方法

### 方式1：使用演示脚本

最简单的方式是运行历史对比演示脚本：

```bash
python historical_comparison_demo.py
```

该脚本会：
1. 生成当前和历史的模拟负荷数据
2. 进行负荷阶段划分
3. 执行历史对比分析
4. 生成可视化图表和文本报告

**输出文件：**
- `/tmp/historical_comparison_demo.png` - 8个子图的综合可视化
- `/tmp/historical_comparison_report.txt` - 详细的文本分析报告

### 方式2：在代码中调用

```python
from train_household_forecast import compare_with_historical_stages

# 准备数据
current_segments = [...]  # 当前负荷分段
historical_segments = [...]  # 历史负荷分段
current_feat_df = pd.DataFrame(...)  # 当前特征数据
historical_feat_df = pd.DataFrame(...)  # 历史特征数据

# 执行对比分析
comparison = compare_with_historical_stages(
    current_segments, historical_segments,
    current_feat_df, historical_feat_df,
    current_times, historical_times,
    current_load, historical_load
)

# 使用对比结果
stage_count_comparison = comparison['stage_count_comparison']
aligned_stages = comparison['aligned_stages']
significant_differences = comparison['significant_differences']
behavior_explanations = comparison['behavior_explanations']
```

### 方式3：集成到预测系统

在 `train_household_forecast.py` 中调用：

```python
# 获取当前预测的负荷阶段
current_explanations = explain_load_changes(segments, feat_df, pred_times, load_values)

# 如果有历史数据，进行对比
if has_historical_data:
    comparison = compare_with_historical_stages(
        current_segments, historical_segments,
        current_feat_df, historical_feat_df,
        current_times, historical_times,
        current_load, historical_load
    )
    
    # 将对比结果添加到解释中
    current_explanations['historical_comparison'] = comparison
    
    # 生成包含历史对比的报告
    generate_explanation_report(current_explanations, output_path)
```

## 可视化输出

演示脚本生成的可视化包含8个子图：

1. **当前负荷曲线与阶段划分** - 显示当前负荷和识别的阶段
2. **历史负荷曲线与阶段划分** - 显示历史负荷和识别的阶段
3. **阶段数量对比** - 柱状图对比当前和历史的阶段数
4. **对齐阶段负荷对比** - 并排柱状图对比每对对齐阶段的负荷
5. **负荷差异百分比** - 显示每个阶段的负荷变化百分比
6. **显著差异阶段摘要** - 文字描述差异最大的阶段
7. **环境因素对比** - 对比当前和历史的温度变化
8. **行为模式解释** - 总体行为分析的文字说明

## 技术实现

### 阶段对齐算法

```python
# 对每个当前阶段，找到历史中最匹配的阶段
for current_stage in current_segments:
    best_match = None
    best_overlap = 0
    
    for historical_stage in historical_segments:
        # 计算时间重叠
        overlap = calculate_time_overlap(current_stage, historical_stage)
        
        # 计算中心点时间差
        time_diff = calculate_center_time_diff(current_stage, historical_stage)
        
        # 选择最佳匹配
        if overlap > best_overlap or (overlap == best_overlap and time_diff < best_time_diff):
            best_match = historical_stage
            best_overlap = overlap
            best_time_diff = time_diff
    
    aligned_pairs.append((current_stage, best_match))
```

### 差异阈值

- **显著差异阈值：** 负荷变化超过 ±20%
- **环境特征差异阈值：**
  - 温度差异：±5°C
  - 湿度差异：±15%
  - 云量差异：±0.3

### 行为解释规则

基于时间段和负荷变化方向，使用规则引擎生成解释：

| 时间段 | 负荷增加原因 | 负荷减少原因 |
|--------|-------------|-------------|
| 夜间(0-6h) | 就寝时间推迟、夜间使用电器增加 | 就寝时间提前、关闭更多电器 |
| 早高峰(6-9h) | 起床时间提前、早餐准备更复杂 | 外出时间提前、简化早餐准备 |
| 上午(9-12h) | 在家办公、使用更多电器 | 家庭成员外出增加 |
| 午间(12-14h) | 在家用餐、使用厨房电器增加 | 外出用餐、减少厨房电器使用 |
| 下午(14-18h) | 在家时间增加、使用娱乐设备 | 外出时间延长、改善节能习惯 |
| 晚高峰(18-22h) | 回家时间提前、晚餐准备更复杂 | 回家时间推迟、简化晚餐准备 |

## 应用场景

1. **负荷预测验证**
   - 对比预测日与历史同期的负荷模式
   - 识别预测异常并分析原因
   - 提高预测结果的可解释性

2. **用电行为监测**
   - 跟踪用电习惯的长期变化
   - 识别异常用电模式
   - 发现节能机会

3. **能源管理优化**
   - 评估节能措施的效果
   - 分析季节性用电变化
   - 制定个性化的能源管理建议

4. **异常检测**
   - 识别突发的负荷变化
   - 分析异常原因（设备故障、行为变化等）
   - 及时发出警报

## 示例输出解读

### 场景：工作日 vs 周末对比

**阶段数量变化：**
```
当前阶段数: 18 (工作日)
历史阶段数: 24 (周末)
变化: -6 个阶段 (-25.0%)
```

**解释：** 工作日阶段数减少，因为家庭成员白天外出工作，用电模式更简单、更规律。

**显著差异阶段：**
```
阶段 5 (9:00-18:00):
  负荷变化: -1.2 kW (-60%)
  解释：上午到下午时段负荷大幅减少，家庭成员外出工作，只保留必要的待机负荷
```

### 场景：夏季 vs 冬季对比

**显著差异阶段：**
```
阶段 8 (14:00-18:00):
  负荷变化: +2.5 kW (+120%)
  环境差异：温度 +12°C
  解释：下午时段负荷大幅增加，高温导致空调制冷需求显著上升
```

## 限制和注意事项

1. **数据质量要求**
   - 需要完整的负荷时序数据
   - 环境特征数据可选，但有助于提供更准确的解释
   - 时间对齐要求：当前和历史数据应具有相同的时间粒度

2. **对齐精度**
   - 简单的时间重叠匹配可能不适用于所有场景
   - 对于负荷模式差异很大的情况，对齐结果可能不理想
   - 建议用于相似场景的对比（如同一星期几、同一季节）

3. **解释的主观性**
   - 行为解释基于常见的用电模式规则
   - 实际原因可能因家庭而异
   - 建议结合实际情况进行判断

## 未来改进方向

1. **高级对齐算法**
   - 实现动态时间规整（DTW）算法
   - 考虑负荷形状相似性
   - 支持多对一、一对多的阶段匹配

2. **机器学习增强**
   - 使用历史数据训练行为解释模型
   - 自动学习用户特定的用电模式
   - 提供个性化的解释

3. **多时间尺度对比**
   - 支持日、周、月、年多尺度对比
   - 识别长期趋势
   - 季节性模式分析

4. **交互式可视化**
   - 开发Web界面
   - 支持用户自定义对比参数
   - 实时更新分析结果

## 参考文献

1. 时间序列相似性度量方法
2. 动态时间规整（DTW）算法
3. 家庭负荷模式识别
4. 可解释人工智能在能源领域的应用

## 联系方式

如有问题或建议，请通过GitHub Issues反馈。
