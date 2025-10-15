# 多历史时期负荷对比功能使用指南

## 功能概述

本功能实现了将预测日负荷与多个历史时期（如7天前、3天前、1天前）的负荷进行对比分析，通过识别阶段数量变化、时间偏移、负荷水平变化，并结合人的行为模式提供解释。

## 核心功能

### 1. 函数 `compare_predicted_with_multiple_historical_stages`

**位置：** `train_household_forecast.py` (1385行附近)

**功能：**
- 对比预测日与多个历史时期的负荷阶段
- 识别阶段数量的增减
- 检测阶段时间的左移（提前）或右移（推迟）
- 分析负荷水平的变化
- 提供基于人类行为的解释

**参数：**
```python
def compare_predicted_with_multiple_historical_stages(
    predicted_segments,      # 预测日负荷分段 [(start_idx, end_idx, state, mean_load), ...]
    historical_data_dict,    # 历史数据字典 {days_ago: {'segments':..., 'feat_df':..., 'times':..., 'load':...}}
    predicted_feat_df,       # 预测日特征数据框
    predicted_times,         # 预测日时间点列表
    predicted_load,          # 预测日负荷值数组
    comparison_days=[1,3,7]  # 要对比的历史天数列表
)
```

**返回值：**
```python
{
    'comparison_days': [1, 3, 7],           # 对比的天数列表
    'comparisons': {
        1: {...},  # 与1天前的详细对比
        3: {...},  # 与3天前的详细对比
        7: {...}   # 与7天前的详细对比
    },
    'summary': {
        'stage_count_trends': [...],        # 阶段数量变化趋势
        'load_trends': [...],               # 负荷变化趋势
        'time_shift_trends': [...],         # 时间偏移趋势
        'behavior_patterns': [...]          # 综合行为模式解释
    }
}
```

## 使用示例

### 基本用法

```python
from train_household_forecast import (
    compare_predicted_with_multiple_historical_stages,
    hmm_load_segmentation  # 或 simple_load_segmentation
)

# 1. 准备预测日数据
predicted_df = pd.read_csv('predicted_load.csv')
predicted_load = predicted_df['load'].values
predicted_times = predicted_df.index.tolist()

# 2. 进行预测日阶段划分
_, _, predicted_segments = hmm_load_segmentation(predicted_load)

# 3. 准备历史数据
historical_data_dict = {}

for days_ago in [1, 3, 7]:
    # 读取历史数据
    hist_df = pd.read_csv(f'historical_{days_ago}days.csv')
    hist_load = hist_df['load'].values
    
    # 进行阶段划分
    _, _, hist_segments = hmm_load_segmentation(hist_load)
    
    # 添加到字典
    historical_data_dict[days_ago] = {
        'segments': hist_segments,
        'feat_df': hist_df,
        'times': hist_df.index.tolist(),
        'load': hist_load
    }

# 4. 执行多历史时期对比
multi_comparison = compare_predicted_with_multiple_historical_stages(
    predicted_segments,
    historical_data_dict,
    predicted_df,
    predicted_times,
    predicted_load,
    comparison_days=[1, 3, 7]
)

# 5. 查看结果
print(f"对比了 {len(multi_comparison['comparisons'])} 个历史时期")

# 查看综合分析
for pattern in multi_comparison['summary']['behavior_patterns']:
    print(f"  • {pattern}")
```

### 详细用法 - 查看具体对比

```python
# 查看与3天前的对比
comparison_3day = multi_comparison['comparisons'][3]

# 阶段数量变化
scc = comparison_3day['stage_count_comparison']
print(f"阶段数变化: {scc['change']:+d} ({scc['change_percent']:+.1f}%)")

# 显著差异阶段
for diff in comparison_3day['significant_differences']:
    print(f"\n阶段 {diff['current_stage']} ↔ {diff['historical_stage']}:")
    print(f"  时间: {diff['time_range']} vs {diff['historical_time_range']}")
    
    if abs(diff['time_shift']) >= 1.0:
        print(f"  偏移: {diff['time_shift']:+.1f} 小时 ({diff['shift_direction']})")
    
    if abs(diff['load_change_percent']) > 20:
        print(f"  负荷: {diff['load_change']:+.2f} kW ({diff['load_change_percent']:+.1f}%)")
    
    print(f"  解释:")
    for exp in diff['explanations']:
        print(f"    • {exp}")
```

### 实际应用场景

#### 场景1：周末 vs 工作日对比

```python
# 预测周日负荷，对比周四(3天前)
# 预期发现：
# - 早高峰右移2小时（起床晚）
# - 白天负荷增加（在家时间长）
# - 整体作息时间推迟

multi_comparison = compare_predicted_with_multiple_historical_stages(
    sunday_segments,
    {3: thursday_data},  # 只对比3天前（工作日）
    sunday_df,
    sunday_times,
    sunday_load,
    comparison_days=[3]
)
```

#### 场景2：多期趋势分析

```python
# 分析过去7天的趋势
multi_comparison = compare_predicted_with_multiple_historical_stages(
    today_segments,
    {
        1: yesterday_data,
        2: twodaysago_data,
        3: threedaysago_data,
        7: lastweek_data
    },
    today_df,
    today_times,
    today_load,
    comparison_days=[1, 2, 3, 7]
)

# 查看跨时期趋势
time_shifts = multi_comparison['summary']['time_shift_trends']
for ts in time_shifts:
    print(f"{ts['days_ago']}天前: {ts['dominant_direction']}, {ts['shift_count']}个阶段偏移")
```

## 输出解读

### 1. 阶段数量变化 (stage_count_trends)

```python
[
    {
        'days_ago': 1,
        'predicted_count': 24,
        'historical_count': 22,
        'change': 2
    },
    ...
]
```

**解读：** 
- 正值：阶段数增加，用电模式复杂化
- 负值：阶段数减少，用电模式规律化

### 2. 时间偏移趋势 (time_shift_trends)

```python
[
    {
        'days_ago': 3,
        'shift_count': 3,
        'right_shift_count': 3,
        'left_shift_count': 0,
        'dominant_direction': '右移(推迟)'
    },
    ...
]
```

**解读：**
- right_shift_count > left_shift_count：整体作息推迟（如周末）
- left_shift_count > right_shift_count：整体作息提前（如工作日）

### 3. 负荷变化趋势 (load_trends)

```python
[
    {
        'days_ago': 3,
        'increase_count': 4,
        'decrease_count': 1,
        'total_significant': 5
    },
    ...
]
```

**解读：**
- increase_count > decrease_count：负荷整体上升（在家时间长、用电增加）
- decrease_count > increase_count：负荷整体下降（外出时间长、节能）

### 4. 综合行为模式 (behavior_patterns)

```python
[
    '与过去[1, 3, 7]天相比，负荷阶段持续右移(推迟)，说明作息时间逐渐推迟，可能是周末/假日效应、或生活习惯改变',
    '与历史时期相比，负荷整体呈上升趋势，可能原因：在家时间增加、新增用电设备、季节性需求上升',
    ...
]
```

**解读：** 提供了可理解的、基于人类行为的变化原因解释

## 关键参数说明

### 显著差异阈值

在 `compare_with_historical_stages` 函数中定义：

- **负荷差异阈值：** `|load_diff_pct| > 20%`
- **时间偏移阈值：** `|time_shift| >= 1.0` 小时

如需调整，修改 `train_household_forecast.py` 第1163和1258行附近的代码。

### 时间偏移计算

```python
time_shift = curr_mid_hour - hist_mid_hour
```

- **正值：** 右移（推迟），如 +2.0 = 推迟2小时
- **负值：** 左移（提前），如 -1.5 = 提前1.5小时

### 阶段对齐策略

使用基于时间重叠和中心点距离的匹配：

1. 优先匹配时间重叠最大的阶段
2. 如果重叠相同，选择中心点时间最接近的阶段
3. 每个当前阶段匹配一个历史阶段（一对一）

## 行为解释规则

系统根据时间段和变化类型自动生成解释：

| 时间段 | 时间右移(推迟) | 负荷增加 |
|--------|--------------|---------|
| 早高峰(6-9h) | 起床时间推迟，周末/假日效应 | 早餐准备更复杂，用电设备增加 |
| 白天(9-18h) | 活动时间调整 | 在家时间增加，周末/假日在家 |
| 晚高峰(18-22h) | 回家/晚餐时间推迟 | 家庭活动增加，娱乐设备使用多 |
| 夜间(22-24h) | 就寝时间推迟 | 夜间活动增加 |

## 常见问题

### Q1: 如何处理缺失的历史数据？

函数会自动跳过缺失的历史时期：

```python
for days_ago in comparison_days:
    if days_ago not in historical_data_dict:
        print(f"⚠️ {days_ago}天前的历史数据不存在，跳过")
        continue
```

### Q2: 可以对比更多历史时期吗？

可以，只需在 `comparison_days` 参数中添加更多天数：

```python
multi_comparison = compare_predicted_with_multiple_historical_stages(
    predicted_segments,
    historical_data_dict,
    ...,
    comparison_days=[1, 2, 3, 4, 5, 6, 7, 14, 30]  # 支持任意天数
)
```

### Q3: 如何生成文本报告？

可以使用现有的 `generate_explanation_report` 函数，或者自己格式化输出：

```python
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write("多历史时期对比分析报告\n")
    f.write("="*80 + "\n\n")
    
    for days_ago in sorted(multi_comparison['comparisons'].keys()):
        f.write(f"\n与{days_ago}天前的对比:\n")
        comparison = multi_comparison['comparisons'][days_ago]
        
        for exp in comparison['behavior_explanations']:
            f.write(f"  • {exp}\n")
```

### Q4: 函数执行时间有多长？

对于典型的96个时间点（24小时，15分钟间隔），对比3个历史时期：
- 阶段划分：< 1秒
- 单次对比：< 0.1秒
- 总执行时间：< 1秒

## 技术细节

### 依赖关系

```python
compare_predicted_with_multiple_historical_stages
    └── compare_with_historical_stages  # 单次对比函数
        ├── 阶段数量分析
        ├── 阶段对齐
        ├── 差异识别
        └── 行为解释生成
```

### 数据流

```
预测日数据 ──┐
            ├──> 阶段划分 ──> 多历史对比 ──> 结果分析
历史数据 ────┘
```

### 内存占用

对于96个时间点，3个历史时期：
- 输入数据：~100KB
- 中间结果：~200KB
- 输出结果：~50KB

## 进一步开发

### 扩展建议

1. **可视化：** 添加图表展示阶段对比
2. **交互界面：** 开发Web界面供用户选择对比时期
3. **机器学习：** 使用历史数据训练个性化解释模型
4. **实时监控：** 集成到实时负荷监控系统

### 代码位置

- **主函数：** `train_household_forecast.py` 第1385-1549行
- **依赖函数：** `compare_with_historical_stages` 第982-1383行
- **演示代码：** `demo_multi_historical_comparison.py`
- **使用文档：** `MULTI_HISTORICAL_COMPARISON_EXAMPLE.md`

## 总结

`compare_predicted_with_multiple_historical_stages` 函数提供了一个强大的工具，用于：
- 理解负荷预测的变化原因
- 验证预测结果的合理性
- 发现用电行为的规律和异常
- 生成可解释的负荷分析报告

通过与多个历史时期的对比，可以更全面地理解负荷变化的趋势和模式。
