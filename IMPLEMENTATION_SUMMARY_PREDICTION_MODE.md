# 实现总结：预测模式中的多历史时期负荷对比分析功能

## 需求描述

根据问题陈述：

> 通过将预测日负荷与历史负荷（7/3/1天）的划分阶段进行对比（结合负荷和环境特征、是否工作日等），比较阶段数量的增多或减少，阶段的左移或右移，阶段中负荷的增加减少，结合人的行为进行解释，如因为是周末，起床时间推迟，早高峰后移2小时，用电设备增多，负荷提高；以此为例，讲清楚与历史阶段相比预测日负荷阶段变化而原因。请在预测模式实现这个功能

## 实现方案

### 1. 核心功能集成位置

功能已集成到 **`train_household_forecast.py`** 的 **预测模式（predict_mode）** 中。

具体集成点：
- **`plot_single_day_prediction` 函数** - 在预测图生成后，自动触发多历史时期对比分析

### 2. 新增核心函数

#### 2.1 `prepare_historical_data_for_comparison`

**功能：** 准备历史数据用于对比分析

**位置：** `train_household_forecast.py` 第1552行

**参数：**
- `ts` - 原始时间序列数据
- `feat_df` - 特征数据框
- `target_date` - 预测目标日期
- `comparison_days` - 对比的历史天数列表（默认[1, 3, 7]）

**返回：**
- `historical_data_dict` - 历史数据字典，包含各历史时期的segments、特征、时间、负荷

**实现逻辑：**
1. 对每个历史天数（1, 3, 7天前）：
   - 计算历史日期
   - 从时间序列中提取该日期的数据
   - 使用HMM对历史负荷进行阶段划分
   - 提取该日期的环境特征
   - 将数据存入字典

2. 错误处理：
   - 如果某天历史数据不存在，跳过并提示
   - 如果HMM失败，降级到简单分段方法

#### 2.2 `generate_multi_historical_comparison_report`

**功能：** 生成多历史时期对比分析报告（文本格式）

**位置：** `train_household_forecast.py` 第1625行

**参数：**
- `multi_comparison` - 对比分析结果字典
- `output_path` - 报告保存路径

**输出内容：**
1. 阶段数量变化趋势
2. 与各历史时期的详细对比
   - 阶段数量对比
   - 显著差异阶段
   - 时间偏移分析
   - 负荷变化分析
   - 行为解释
3. 跨时期行为模式总结
4. 时间偏移趋势统计
5. 负荷变化趋势统计

### 3. 集成到预测流程

#### 3.1 触发时机

在 `plot_single_day_prediction` 函数中，在完成以下步骤后自动触发：
1. ✅ 预测值生成
2. ✅ HMM阶段划分
3. ✅ 可解释性分析
4. **🆕 多历史时期对比分析** ← 新增

#### 3.2 集成代码（第2819-2866行）

```python
# 🆕 多历史时期对比分析
print("\n🔍 开始多历史时期负荷对比分析...")
multi_comparison = None
multi_comparison_report_path = None

try:
    # 准备历史数据（1, 3, 7天前）
    historical_data_dict = prepare_historical_data_for_comparison(
        ts, feat_df, pred_date, comparison_days=[1, 3, 7]
    )
    
    if historical_data_dict:
        # 执行多历史时期对比分析
        multi_comparison = compare_predicted_with_multiple_historical_stages(
            segments,
            historical_data_dict,
            feat_df.loc[day_index] if len(day_index) > 0 else feat_df,
            pred_times,
            pred_resampled.values,
            comparison_days=[1, 3, 7]
        )
        
        # 保存多历史时期对比分析报告
        multi_comparison_report_path = os.path.join(
            out_dir, 
            f'multi_historical_comparison_{pred_date_obj.strftime("%Y%m%d")}.txt'
        )
        generate_multi_historical_comparison_report(
            multi_comparison, 
            multi_comparison_report_path
        )
        
        print(f"✅ 多历史时期对比分析完成，报告已保存")
        
        # 打印简要对比结果
        print("\n📊 多历史时期对比摘要:")
        if multi_comparison['summary'].get('behavior_patterns'):
            for pattern in multi_comparison['summary']['behavior_patterns'][:3]:
                print(f"   • {pattern}")
    else:
        print("⚠️ 无足够历史数据进行多历史时期对比分析")
        
except Exception as e:
    print(f"⚠️ 多历史时期对比分析失败: {e}")
```

#### 3.3 结果保存

预测模式现在会额外保存两个文件：

1. **文本报告** - `multi_historical_comparison_YYYYMMDD.txt`
2. **JSON数据** - `multi_historical_comparison_YYYYMMDD.json`

保存位置：`output/analysis/{model_name}/predictions/`

## 功能特点

### 1. 阶段数量对比

对比预测日与各历史时期的阶段数量，分析：
- 阶段数增加/减少的数量和百分比
- 趋势判断（增加/减少/不变）
- 基于行为的解释原因

**示例：**
```
与3天前相比:
  预测日阶段数: 5
  历史阶段数: 4
  变化: +1 个阶段 (+25.0%)
  
原因分析:
  负荷阶段数增加1个，可能原因：
    1. 用电行为更加多样化，出现更多负荷切换
    2. 家庭成员活动模式发生变化
    3. 新增用电设备或改变使用习惯
```

### 2. 时间偏移识别

识别阶段的时间偏移（≥1小时为显著偏移）：
- **右移（推迟）** - 正值，如 +2.1小时
- **左移（提前）** - 负值，如 -1.5小时

根据时间段提供针对性解释：
- 早高峰（6-9h）：起床时间变化
- 午间（12-14h）：用餐时间变化
- 晚高峰（18-22h）：回家/晚餐时间变化
- 夜间（22-24h/0-6h）：就寝时间变化

**示例：**
```
⏰ 时间偏移: +2.1 小时 (右移/推迟)
🔍 行为解释:
  • 早高峰阶段时间推迟约2.1小时，可能是：因为周末/假日导致起床时间推迟、
    或作息时间调整
```

### 3. 负荷水平变化

识别显著负荷差异（>20%）：
- 计算负荷变化量和百分比
- 判断变化类型（增加/减少）
- 基于时间段提供行为解释

**示例：**
```
📊 负荷变化:
  预测日负荷: 1.8165
  历史负荷: 0.8041
  变化量: +1.0124 (+125.9%)
  变化类型: 增加
🔍 行为解释:
  • 上午时段负荷增加，可能是：在家办公、使用更多电器、或家庭成员未外出
```

### 4. 跨时期趋势分析

分析3个历史时期的整体趋势：
- 阶段数量变化的一致性
- 时间偏移方向的一致性
- 负荷水平变化的整体趋势

识别模式：
- 持续增加/减少
- 持续右移/左移
- 波动/不稳定

**示例：**
```
【跨时期行为模式总结】
• 与过去[1, 3, 7]天相比，负荷阶段持续右移(推迟)，说明作息时间逐渐推迟，
  可能是周末/假日效应、或生活习惯改变
• 与历史时期相比，负荷整体呈上升趋势，可能原因：在家时间增加、
  新增用电设备、季节性需求上升
```

## 技术实现细节

### 1. 阶段对齐算法

使用现有的 `compare_with_historical_stages` 函数进行阶段对齐：
- 基于时间重叠度匹配最相似的阶段
- 计算时间偏移（阶段中心点的差异）
- 计算负荷差异

### 2. 行为解释规则

#### 时间段定义
- 夜间：0-6h
- 早高峰：6-9h
- 上午：9-12h
- 午间：12-14h
- 下午：14-18h
- 晚高峰：18-22h
- 深夜：22-24h

#### 解释策略
根据时间段和负荷变化方向，从预定义的规则表中选择合适的解释。

示例规则：
- 早高峰 + 负荷增加 → "起床时间提前、早餐准备更复杂"
- 早高峰 + 时间推迟 → "周末/假日导致起床时间推迟"
- 白天 + 负荷增加 → "在家时间增加、周末/假日在家"

### 3. 环境特征对比

对比预测日与历史日期的环境特征差异：
- 温度差异（> ±5°C）
- 湿度差异（> ±15%）
- 云量差异（> ±0.3）

影响解释：
- 温度差异 → 影响空调/暖气使用
- 湿度差异 → 影响除湿设备使用
- 云量差异 → 影响照明需求

## 测试验证

### 测试文件
`test_prediction_mode_multi_comparison.py`

### 测试内容
1. ✅ 历史数据准备功能
2. ✅ 多历史时期对比分析功能
3. ✅ 报告生成功能
4. ✅ 完整的工作流程

### 测试结果
所有测试通过，功能正常工作。

## 输出示例

完整的报告示例请参考：
- 测试生成的报告：`/tmp/test_multi_historical_comparison_report.txt`
- 文档中的示例：[USAGE_EXAMPLE_PREDICTION_MODE.md](USAGE_EXAMPLE_PREDICTION_MODE.md)

## 使用方法

### 自动触发（推荐）

在预测模式中，功能会自动触发，无需额外配置：

```bash
python train_household_forecast.py
# 选择 "预测模式"
# 选择模型和预测日期
# 系统自动生成多历史时期对比分析报告
```

### 手动调用

如需在代码中手动调用：

```python
from train_household_forecast import (
    prepare_historical_data_for_comparison,
    compare_predicted_with_multiple_historical_stages,
    generate_multi_historical_comparison_report
)

# 1. 准备历史数据
historical_data_dict = prepare_historical_data_for_comparison(
    ts, feat_df, target_date, comparison_days=[1, 3, 7]
)

# 2. 执行对比分析
multi_comparison = compare_predicted_with_multiple_historical_stages(
    predicted_segments,
    historical_data_dict,
    predicted_feat_df,
    predicted_times,
    predicted_load,
    comparison_days=[1, 3, 7]
)

# 3. 生成报告
generate_multi_historical_comparison_report(
    multi_comparison, 
    'output_report.txt'
)
```

## 相关文档

1. **[预测模式中的多历史时期对比分析](PREDICTION_MODE_MULTI_HISTORICAL_COMPARISON.md)** - 功能详细说明
2. **[使用示例](USAGE_EXAMPLE_PREDICTION_MODE.md)** - 完整的使用流程和示例
3. **[多历史时期对比分析示例](MULTI_HISTORICAL_COMPARISON_EXAMPLE.md)** - 输出示例
4. **[实现细节](IMPLEMENTATION_DETAILS.md)** - 技术实现细节

## 总结

### 已完成
✅ 在预测模式中集成多历史时期负荷对比分析功能  
✅ 自动对比预测日与1/3/7天前的历史负荷  
✅ 识别阶段数量变化、时间偏移、负荷水平变化  
✅ 结合人的行为习惯提供解释  
✅ 生成详细的文本报告和JSON数据  
✅ 添加完整的测试和文档  

### 核心价值
- 🎯 帮助理解预测日与历史负荷的差异原因
- 🔍 基于人的行为模式提供直观的解释
- 📊 自动化分析，无需手动对比
- 💾 完整的数据保存，便于后续分析

### 适用场景
- 验证预测合理性
- 分析用电行为变化
- 识别异常用电模式
- 优化能源管理策略
- 支持需求响应决策
