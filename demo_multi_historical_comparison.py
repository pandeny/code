"""
多历史时期负荷对比演示
展示如何将预测日负荷与7/3/1天前的历史负荷进行对比分析

功能演示:
1. 对比预测日与7/3/1天前的阶段数量变化
2. 分析阶段的时间偏移（左移/右移）
3. 分析负荷水平的增减
4. 结合人的行为模式提供解释

示例场景：
- 预测日：2024-01-14 (周日 - 周末模式)
- 1天前：2024-01-13 (周六 - 周末模式)
- 3天前：2024-01-11 (周四 - 工作日模式)  
- 7天前：2024-01-07 (周日 - 周末模式)

预期结果：
- 与3天前(工作日)相比，早高峰右移(推迟)约2小时，因为周末起床时间晚
- 与3天前相比，白天负荷增加，因为周末在家时间长
- 与1天前和7天前(周末)相比，模式相似但略有波动
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_load_data(date_str, scenario='weekday', seed=None):
    """生成负荷数据"""
    if seed is not None:
        np.random.seed(seed)
    
    n_points = 96  # 24小时 * 4个15分钟
    times = pd.date_range(date_str, periods=n_points, freq='15min')
    hours = np.arange(n_points) / 4
    load = np.zeros(n_points)
    
    # 根据场景生成不同的负荷模式
    for i, h in enumerate(hours):
        if scenario == 'weekday':
            # 工作日模式：早高峰早，白天负荷低（外出工作）
            if h < 6:
                base = 0.5
            elif h < 8:  # 早高峰 6-8h
                progress = (h - 6) / 2
                base = 0.5 + progress * 2.0
            elif h < 9:
                base = 2.5
            elif h < 18:  # 白天外出
                base = 0.8
            elif h < 22:  # 晚高峰
                progress = (h - 18) / 4
                base = 0.8 + progress * 2.5
            else:
                progress = (h - 22) / 2
                base = 3.3 - progress * 2.5
        
        elif scenario == 'weekend':
            # 周末模式：早高峰晚（起床晚），白天负荷高（在家）
            if h < 8:  # 睡得晚
                base = 0.5
            elif h < 10:  # 早高峰推迟到 8-10h
                progress = (h - 8) / 2
                base = 0.5 + progress * 2.5
            elif h < 11:
                base = 3.0
            elif h < 18:  # 白天在家，负荷较高
                base = 1.8
            elif h < 23:  # 晚高峰也推迟
                progress = (h - 18) / 5
                base = 1.8 + progress * 2.0
            else:
                progress = (h - 23) / 1
                base = 3.8 - progress * 2.8
        
        load[i] = base + np.random.normal(0, 0.08)
    
    # 确保负荷为正值
    load = np.maximum(load, 0.3)
    
    # 生成环境特征
    base_temp = {'weekday': 15, 'weekend': 18}.get(scenario, 15)
    temperature = base_temp + 8 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 1, n_points)
    humidity = 60 + 12 * np.cos((hours - 3) * np.pi / 12) + np.random.normal(0, 2, n_points)
    cloudCover = np.clip(0.3 + 0.3 * np.sin(hours * np.pi / 24) + np.random.normal(0, 0.1, n_points), 0, 1)
    
    df = pd.DataFrame({
        'time': times,
        'load': load,
        'temperature_current': temperature,
        'humidity_current': humidity,
        'cloudCover_current': cloudCover,
        'hour': hours
    })
    
    df = df.set_index('time')
    return df

def simple_segmentation(load_values, n_segments=4):
    """
    简单的负荷分段方法（基于分位数）
    增强版：使用时间特征的正余弦编码让时间成为连续的，并检测负荷峰值/波动区作为窗口划分阶段
    """
    load_values = np.array(load_values)
    n = len(load_values)
    
    # 构建时间特征（正余弦编码让时间成为连续的）
    features = []
    time_features = []
    for i in range(n):
        hour = (i * 0.25) % 24  # 假设15分钟间隔
        time_features.append([
            np.sin(2 * np.pi * hour / 24),  # 小时的正弦编码
            np.cos(2 * np.pi * hour / 24),  # 小时的余弦编码
            np.sin(2 * np.pi * (i % 96) / 96),  # 日内位置编码
            np.cos(2 * np.pi * (i % 96) / 96)
        ])
    features.append(np.array(time_features))
    time_features = features[0]  # 提取时间特征数组
    
    features.append(np.array(time_features))
    time_features = features[0]  # 提取时间特征数组
    
    # 检测负荷峰值/波动区域
    from scipy.ndimage import median_filter
    smoothed_load = median_filter(load_values.astype(float), size=3)
    window_size = 8  # 2小时窗口
    peak_zones = []
    fluctuation_zones = []
    
    for i in range(window_size, n - window_size):
        window = smoothed_load[i-window_size:i+window_size]
        center_val = smoothed_load[i]
        
        # 峰值检测：当前点是局部最大值
        if center_val == np.max(window) and center_val > np.percentile(smoothed_load, 75):
            peak_zones.append(i)
        
        # 波动检测：窗口内标准差较大
        window_std = np.std(window)
        if window_std > np.std(smoothed_load) * 0.8:
            fluctuation_zones.append(i)
    
    # 合并相邻的峰值/波动区域，形成窗口边界
    def merge_zones(zones, min_gap=6):
        if not zones:
            return []
        zones = sorted(set(zones))
        merged = [zones[0]]
        for z in zones[1:]:
            if z - merged[-1] < min_gap:
                continue
            merged.append(z)
        return merged
    
    peak_boundaries = merge_zones(peak_zones)
    fluctuation_boundaries = merge_zones(fluctuation_zones)
    important_boundaries = sorted(set(peak_boundaries + fluctuation_boundaries))
    
    # 归一化负荷值
    load_normalized = (load_values - load_values.min()) / (load_values.max() - load_values.min() + 1e-10)
    
    quantiles = np.linspace(0, 1, n_segments + 1)
    thresholds = np.quantile(load_values, quantiles)
    states = np.digitize(load_values, thresholds[1:-1])
    
    # 在重要边界处强制分割，确保峰值/波动区作为独立阶段
    for boundary in important_boundaries:
        if 0 < boundary < n-1:
            load_change = abs(load_normalized[boundary] - load_normalized[boundary-1])
            if load_change > 0.15:
                states[boundary] = max(0, states[boundary])
    
    # 使用时间特征优化状态边界
    for i in range(1, n - 1):
        if states[i] != states[i-1]:
            # 如果在重要边界附近，保持分割
            near_boundary = any(abs(i - b) < 3 for b in important_boundaries)
            if near_boundary:
                continue
            
            # 计算时间相似度
            time_sim_prev = np.dot(time_features[i], time_features[i-1])
            
            # 如果时间特征变化不显著，且负荷差异小，则合并状态
            if time_sim_prev > 0.95 and abs(load_normalized[i] - load_normalized[i-1]) < 0.1:
                states[i] = states[i-1]
    
    state_means = []
    for state in range(n_segments):
        state_mask = (states == state)
        if np.any(state_mask):
            state_means.append(np.mean(load_values[state_mask]))
        else:
            state_means.append(np.mean(load_values))
    
    state_means = np.array(state_means)
    
    segments = []
    current_state = states[0]
    start_idx = 0
    
    for i in range(1, len(states)):
        if states[i] != current_state:
            end_idx = i - 1
            segment_load = np.mean(load_values[start_idx:i])
            segments.append((start_idx, end_idx, current_state, segment_load))
            start_idx = i
            current_state = states[i]
    
    segment_load = np.mean(load_values[start_idx:])
    segments.append((start_idx, len(states) - 1, current_state, segment_load))
    
    return states, state_means, segments

print("="*80)
print("多历史时期负荷对比分析演示")
print("="*80)
print("\n场景说明：")
print("  预测日：2024-01-14 (周日 - 周末模式)")
print("  历史数据：")
print("    1天前 (2024-01-13, 周六 - 周末模式)")
print("    3天前 (2024-01-11, 周四 - 工作日模式)")
print("    7天前 (2024-01-07, 周日 - 周末模式)")
print("="*80)

# 生成数据
print("\n▶ 生成预测日负荷数据 (周末模式)...")
predicted_df = generate_load_data('2024-01-14', scenario='weekend', seed=42)
predicted_load = predicted_df['load'].values

print("▶ 生成历史负荷数据...")
hist_1day_df = generate_load_data('2024-01-13', scenario='weekend', seed=43)
hist_3day_df = generate_load_data('2024-01-11', scenario='weekday', seed=44)
hist_7day_df = generate_load_data('2024-01-07', scenario='weekend', seed=45)

# 进行负荷阶段划分
print("▶ 进行负荷阶段划分...")
_, _, predicted_segments = simple_segmentation(predicted_load, n_segments=5)
_, _, hist_1day_segments = simple_segmentation(hist_1day_df['load'].values, n_segments=5)
_, _, hist_3day_segments = simple_segmentation(hist_3day_df['load'].values, n_segments=5)
_, _, hist_7day_segments = simple_segmentation(hist_7day_df['load'].values, n_segments=5)

print(f"  预测日阶段数: {len(predicted_segments)}")
print(f"  1天前阶段数: {len(hist_1day_segments)}")
print(f"  3天前阶段数: {len(hist_3day_segments)}")
print(f"  7天前阶段数: {len(hist_7day_segments)}")

# 准备对比数据
historical_data_dict = {
    1: {
        'segments': hist_1day_segments,
        'feat_df': hist_1day_df,
        'times': hist_1day_df.index.tolist(),
        'load': hist_1day_df['load'].values
    },
    3: {
        'segments': hist_3day_segments,
        'feat_df': hist_3day_df,
        'times': hist_3day_df.index.tolist(),
        'load': hist_3day_df['load'].values
    },
    7: {
        'segments': hist_7day_segments,
        'feat_df': hist_7day_df,
        'times': hist_7day_df.index.tolist(),
        'load': hist_7day_df['load'].values
    }
}

print("\n▶ 执行多历史时期对比分析...")
print("  (正在加载 compare_predicted_with_multiple_historical_stages 函数...)")

# 动态导入函数
try:
    sys.path.insert(0, os.path.dirname(__file__))
    
    # 临时处理tensorflow导入
    import builtins
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'tensorflow':
            import types
            tf = types.ModuleType('tensorflow')
            tf.random = types.ModuleType('random')
            tf.random.set_seed = lambda x: None
            return tf
        return original_import(name, *args, **kwargs)
    
    builtins.__import__ = mock_import
    
    from train_household_forecast import compare_predicted_with_multiple_historical_stages
    
    builtins.__import__ = original_import
    
    print("  ✓ 成功导入函数")
    
    # 执行多历史时期对比
    multi_comparison = compare_predicted_with_multiple_historical_stages(
        predicted_segments,
        historical_data_dict,
        predicted_df,
        predicted_df.index.tolist(),
        predicted_load,
        comparison_days=[1, 3, 7]
    )
    
    # 打印对比结果
    print("\n" + "="*80)
    print("对比分析结果")
    print("="*80)
    
    # 1. 阶段数量变化总结
    print("\n【1】阶段数量变化趋势:")
    print("-"*80)
    for sc in multi_comparison['summary']['stage_count_trends']:
        days_ago = sc['days_ago']
        change = sc['change']
        change_pct = (change / sc['historical_count'] * 100) if sc['historical_count'] > 0 else 0
        print(f"\n与{days_ago}天前相比:")
        print(f"  预测日阶段数: {sc['predicted_count']}")
        print(f"  历史阶段数: {sc['historical_count']}")
        print(f"  变化: {change:+d} 个阶段 ({change_pct:+.1f}%)")
    
    # 2. 详细的逐期对比
    for days_ago in sorted(multi_comparison['comparisons'].keys()):
        print(f"\n{'='*80}")
        print(f"【2-{days_ago}】与{days_ago}天前的详细对比")
        print("="*80)
        
        comparison = multi_comparison['comparisons'][days_ago]
        
        # 显著差异阶段
        sig_diffs = comparison.get('significant_differences', [])
        if sig_diffs:
            print(f"\n识别出{len(sig_diffs)}个差异显著的阶段:\n")
            
            for diff in sig_diffs[:5]:  # 只显示前5个
                print(f"阶段 {diff['current_stage']} (预测日) ↔ 阶段 {diff['historical_stage']} ({days_ago}天前):")
                print(f"  时间范围: {diff['time_range']} (预测) vs {diff['historical_time_range']} (历史)")
                
                # 时间偏移
                if abs(diff['time_shift']) >= 1.0:
                    print(f"  ⏰ 时间偏移: {diff['time_shift']:+.1f} 小时 ({diff['shift_direction']})")
                
                # 负荷变化
                if abs(diff['load_change_percent']) > 20:
                    print(f"  ⚡ 负荷变化: {diff['load_change']:+.2f} kW ({diff['load_change_percent']:+.1f}%)")
                
                # 行为解释
                if diff['explanations']:
                    print(f"  📝 解释:")
                    for exp in diff['explanations'][:2]:  # 只显示前2个
                        print(f"     • {exp}")
                print()
        
        # 行为模式总结
        behavior_exps = comparison.get('behavior_explanations', [])
        if behavior_exps:
            print(f"\n行为模式总结 (与{days_ago}天前):")
            for exp in behavior_exps[:3]:  # 只显示前3个
                print(f"  • {exp}")
    
    # 3. 跨时期的综合分析
    print("\n" + "="*80)
    print("【3】跨时期综合行为模式分析")
    print("="*80)
    
    patterns = multi_comparison['summary']['behavior_patterns']
    if patterns:
        print()
        for i, pattern in enumerate(patterns, 1):
            print(f"{i}. {pattern}")
    
    # 4. 时间偏移趋势
    print("\n【4】时间偏移趋势:")
    print("-"*80)
    time_shifts = multi_comparison['summary']['time_shift_trends']
    if time_shifts:
        for ts in time_shifts:
            print(f"\n与{ts['days_ago']}天前相比:")
            print(f"  有{ts['shift_count']}个阶段发生时间偏移")
            print(f"  右移(推迟): {ts['right_shift_count']}个阶段")
            print(f"  左移(提前): {ts['left_shift_count']}个阶段")
            print(f"  主导方向: {ts['dominant_direction']}")
    else:
        print("  各阶段时间偏移不显著")
    
    # 5. 负荷变化趋势
    print("\n【5】负荷变化趋势:")
    print("-"*80)
    load_trends = multi_comparison['summary']['load_trends']
    if load_trends:
        for lt in load_trends:
            print(f"\n与{lt['days_ago']}天前相比:")
            print(f"  差异显著的阶段: {lt['total_significant']}个")
            print(f"  负荷增加: {lt['increase_count']}个阶段")
            print(f"  负荷减少: {lt['decrease_count']}个阶段")
    else:
        print("  负荷变化不显著")
    
    # 保存报告
    output_path = '/tmp/multi_historical_comparison_report.txt'
    print(f"\n▶ 保存详细报告到: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("预测日负荷与多历史时期对比分析报告\n")
        f.write("="*80 + "\n\n")
        f.write("报告生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        f.write("场景说明：\n")
        f.write("  预测日：2024-01-14 (周日 - 周末模式)\n")
        f.write("  1天前：2024-01-13 (周六 - 周末模式)\n")
        f.write("  3天前：2024-01-11 (周四 - 工作日模式)\n")
        f.write("  7天前：2024-01-07 (周日 - 周末模式)\n\n")
        
        # 写入完整结果
        for days_ago in sorted(multi_comparison['comparisons'].keys()):
            f.write(f"\n{'='*80}\n")
            f.write(f"与{days_ago}天前的详细对比\n")
            f.write("="*80 + "\n")
            
            comparison = multi_comparison['comparisons'][days_ago]
            sig_diffs = comparison.get('significant_differences', [])
            
            if sig_diffs:
                f.write(f"\n差异显著的阶段 ({len(sig_diffs)}个):\n\n")
                for diff in sig_diffs:
                    f.write(f"阶段 {diff['current_stage']} ↔ 阶段 {diff['historical_stage']}:\n")
                    f.write(f"  时间范围: {diff['time_range']} vs {diff['historical_time_range']}\n")
                    if abs(diff['time_shift']) >= 0.5:
                        f.write(f"  时间偏移: {diff['time_shift']:+.1f} 小时\n")
                    if abs(diff['load_change_percent']) > 10:
                        f.write(f"  负荷变化: {diff['load_change']:+.2f} kW ({diff['load_change_percent']:+.1f}%)\n")
                    f.write(f"  解释:\n")
                    for exp in diff['explanations']:
                        f.write(f"    • {exp}\n")
                    f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("综合行为模式分析\n")
        f.write("="*80 + "\n\n")
        for i, pattern in enumerate(patterns, 1):
            f.write(f"{i}. {pattern}\n")
    
    print("  ✓ 报告已保存")
    
    print("\n" + "="*80)
    print("演示完成！")
    print("="*80)
    print("\n主要发现：")
    if patterns:
        for pattern in patterns[:3]:
            print(f"  • {pattern}")
    
except Exception as e:
    print(f"\n❌ 执行失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n说明：")
    print("  本演示需要 train_household_forecast.py 中的函数")
    print("  如果导入失败，请确保该文件存在且可访问")
