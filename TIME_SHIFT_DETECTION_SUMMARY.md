# 时间偏移检测功能实现总结
# Time Shift Detection Implementation Summary

## 功能概述 (Feature Overview)

本次实现为历史负荷对比分析功能添加了**阶段时间偏移检测**能力，能够识别和解释负荷阶段在时间轴上的左移（提前）或右移（推迟）现象。

This implementation adds **stage time shift detection** capability to the historical load comparison feature, enabling identification and explanation of leftward (earlier) or rightward (later) movement of load stages on the time axis.

## 实现的功能 (Implemented Features)

### 1. 时间偏移计算 (Time Shift Calculation)
- 计算每个对齐阶段的中心点时间差
- 正值表示右移（推迟），负值表示左移（提前）
- 单位：小时

### 2. 显著偏移检测 (Significant Shift Detection)
- 检测标准：|时间偏移| >= 1.0 小时
- 与负荷差异检测（>20%）并行工作
- 任一条件满足即标记为显著差异阶段

### 3. 分阶段解释 (Stage-Specific Explanations)
根据阶段类型提供针对性解释：

#### 早高峰时段 (6-9h)
- 推迟：周末/假日起床时间推迟、作息调整
- 提前：工作日作息提前、早起习惯养成

#### 午间时段 (12-14h)
- 推迟：用餐时间推迟、午休习惯改变
- 提前：用餐时间提前

#### 晚高峰时段 (18-22h)
- 推迟：下班/回家时间推迟、晚餐时间调整
- 提前：下班/回家时间提前

#### 夜间时段 (22h-6h)
- 推迟：就寝时间推迟、夜间活动增加
- 提前：就寝时间提前

### 4. 整体模式分析 (Overall Pattern Analysis)
统计和分析：
- 显著偏移阶段数量
- 右移阶段数量 vs 左移阶段数量
- 整体偏移趋势（右移为主/左移为主/混合）
- 可能的原因分析

## 示例输出 (Example Output)

### 逐阶段分析
```
当前阶段2 ↔ 历史阶段2:
  时间范围: 8.0h-11.0h (周末) vs 6.0h-9.0h (工作日)
  ⏰ 时间偏移: 2.0 小时 → (右移/推迟)
  负荷水平: 2.80 kW (周末) vs 2.50 kW (工作日)
  负荷差异: +0.30 kW (+12.0%)
```

### 差异显著阶段
```
阶段2 (周末时间: 8.0h-11.0h, 工作日时间: 6.0h-9.0h):
  ⏰ 时间偏移: 2.0 小时 (右移/推迟)
  📊 负荷变化: +0.30 kW (+12.0%)
  💡 行为解释:
      • 早高峰阶段时间推迟约2.0小时，可能是：因为周末/假日导致起床时间推迟、或作息时间调整
      • 早高峰时段负荷增加，可能是：起床时间提前、早餐准备更复杂、或增加了热水器/咖啡机使用
```

### 总体分析
```
时间偏移模式：整体右移(推迟)为主，3个阶段有显著时间偏移（2个右移，1个左移）
偏移原因：可能是周末/假日作息推迟、工作时间调整、或生活习惯改变
整体趋势：负荷增加为主(3个阶段增加，1个阶段减少)
可能原因：家庭活动增加、在家时间延长、新增用电设备、或季节性用电需求变化
```

## 应用场景 (Use Cases)

### 1. 工作日 vs 周末对比
- **典型现象**：早高峰右移2小时（6-9h → 8-11h）
- **解释**：周末睡懒觉，起床时间推迟

### 2. 冬季 vs 夏季对比
- **典型现象**：晚高峰左移1-2小时
- **解释**：冬季天黑早，活动时间提前

### 3. 节假日 vs 常规日对比
- **典型现象**：多个阶段整体右移
- **解释**：假日作息推迟，活动更自由

### 4. 在家办公 vs 外出工作对比
- **典型现象**：上午阶段时间分布变化
- **解释**：工作模式改变，在家时间增加

## 技术实现 (Technical Implementation)

### 核心算法
```python
# 计算时间偏移
curr_mid_hour = (curr_start_hour + curr_end_hour) / 2
hist_mid_hour = (hist_start_hour + hist_end_hour) / 2
time_shift = curr_mid_hour - hist_mid_hour

# 判断偏移方向
if time_shift > 0:
    direction = '右移(推迟)'
elif time_shift < 0:
    direction = '左移(提前)'
else:
    direction = '无偏移'
```

### 检测标准
```python
# 显著差异检测
has_load_diff = abs(load_difference_percent) > 20
has_time_shift = abs(time_shift) >= 1.0

if has_load_diff or has_time_shift:
    # 标记为显著差异阶段
    significant_differences.append(stage)
```

## 测试覆盖 (Test Coverage)

### 单元测试
1. `test_time_shift_detection()` - 右移检测
2. `test_left_shift_detection()` - 左移检测
3. 原有的对齐测试 - 向后兼容性

### 集成测试
- `demo_time_shift.py` - 完整场景演示
- 工作日 vs 周末对比
- 输出完整报告

## 文件修改 (Files Modified)

### 核心实现
1. `train_household_forecast.py`
   - `compare_with_historical_stages()` 函数
   - `generate_explanation_report()` 函数

2. `historical_comparison_demo.py`
   - `compare_with_historical_stages_standalone()` 函数

### 测试文件
3. `test_time_shift_detection.py` - 新增
4. `demo_time_shift.py` - 新增

### 文档更新
5. `HISTORICAL_COMPARISON_GUIDE.md`

## 向后兼容性 (Backward Compatibility)

✅ 完全向后兼容
- 所有原有测试通过
- 新增字段为可选，不影响现有功能
- 报告格式保持一致，仅添加新信息

## 使用方法 (Usage)

### 运行演示
```bash
python demo_time_shift.py
```

### 运行测试
```bash
python test_time_shift_detection.py
python test_historical_comparison.py
```

### 在代码中使用
```python
from historical_comparison_demo import compare_with_historical_stages_standalone

comparison = compare_with_historical_stages_standalone(
    current_segments,
    historical_segments,
    current_feat_df,
    historical_feat_df,
    current_times,
    historical_times,
    current_load,
    historical_load
)

# 访问时间偏移信息
for stage in comparison['aligned_stages']:
    time_shift = stage['time_shift']
    print(f"Time shift: {time_shift:.2f}h")

# 访问显著差异（含时间偏移）
for diff in comparison['significant_differences']:
    if 'time_shift' in diff:
        print(f"Shift direction: {diff['shift_direction']}")
```

## 总结 (Summary)

本实现完整满足了问题陈述中的要求：

✅ **阶段数量增减** - 原有功能
✅ **阶段的左移或右移** - 本次新增 ⭐
✅ **阶段中负荷的增加减少** - 原有功能
✅ **结合人的行为进行解释** - 增强完善

特别是针对"早高峰后移2小时"这样的场景，现在能够：
1. 自动检测到时间偏移（+2小时）
2. 识别为右移（推迟）
3. 提供基于行为的解释（周末起床时间推迟）
4. 分析整体偏移模式

---

实现日期：2025-10-14
版本：1.0
