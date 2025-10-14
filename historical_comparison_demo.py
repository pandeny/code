"""
负荷阶段历史对比演示脚本

该脚本展示如何使用历史负荷对比功能来：
1. 比较当前与历史负荷的阶段数量变化
2. 逐阶段对齐分析
3. 识别差异显著的阶段
4. 基于人的行为模式解释差异原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# 添加当前目录到路径以导入train_household_forecast中的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_demo_data(date_str, seed=42, scenario='normal'):
    """
    生成演示用的负荷数据和环境特征
    
    参数:
    - date_str: 日期字符串 (YYYY-MM-DD)
    - seed: 随机种子
    - scenario: 场景类型 ('normal', 'high_morning', 'low_afternoon', 'shift_evening')
    """
    np.random.seed(seed)
    n_points = 96  # 24小时 * 4个15分钟
    times = pd.date_range(date_str, periods=n_points, freq='15min')
    
    # 生成基础负荷曲线
    hours = np.arange(n_points) / 4
    load = np.zeros(n_points)
    
    for i, h in enumerate(hours):
        if h < 6:  # 夜间 (0-6h)
            base = 0.5
        elif h < 9:  # 早高峰 (6-9h)
            progress = (h - 6) / 3
            base = 0.5 + progress * 2.0
        elif h < 18:  # 白天 (9-18h)
            base = 1.0
        elif h < 22:  # 晚高峰 (18-22h)
            progress = (h - 18) / 4
            base = 1.0 + progress * 2.5
        else:  # 深夜 (22-24h)
            progress = (h - 22) / 2
            base = 3.5 - progress * 2.7
        
        # 根据场景调整
        if scenario == 'high_morning' and 6 <= h < 9:
            base *= 1.4  # 早高峰增加40%
        elif scenario == 'low_afternoon' and 14 <= h < 18:
            base *= 0.6  # 下午减少40%
        elif scenario == 'shift_evening' and 18 <= h < 22:
            base *= 1.3  # 晚高峰增加30%
        
        load[i] = base + np.random.normal(0, 0.08)
    
    # 确保负荷为正值
    load = np.maximum(load, 0.3)
    
    # 生成环境特征
    base_temp = 15 if scenario == 'normal' else 20  # 不同场景不同基础温度
    temperature = base_temp + 10 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 1, n_points)
    humidity = 60 + 15 * np.cos((hours - 3) * np.pi / 12) + np.random.normal(0, 2, n_points)
    cloudCover = np.clip(0.3 + 0.4 * np.sin(hours * np.pi / 24) + np.random.normal(0, 0.1, n_points), 0, 1)
    
    df = pd.DataFrame({
        'time': times,
        'load': load,
        'load_smooth': load,
        'temperature_current': temperature,
        'humidity_current': humidity,
        'cloudCover_current': cloudCover,
        'hour': hours,
        'target_next': np.roll(load, -1)
    })
    
    df = df.set_index('time')
    return df

def simple_segmentation(load_values, n_segments=4):
    """简单的负荷分段方法（基于分位数）"""
    load_values = np.array(load_values)
    
    # 计算分位数阈值
    quantiles = np.linspace(0, 1, n_segments + 1)
    thresholds = np.quantile(load_values, quantiles)
    
    # 分配状态
    states = np.digitize(load_values, thresholds[1:-1])
    
    # 计算每个状态的平均值
    state_means = []
    for state in range(n_segments):
        state_mask = (states == state)
        if np.any(state_mask):
            state_mean = np.mean(load_values[state_mask])
            state_means.append(state_mean)
        else:
            state_means.append(np.mean(load_values))
    
    state_means = np.array(state_means)
    
    # 识别连续段
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
    
    # 添加最后一段
    segment_load = np.mean(load_values[start_idx:])
    segments.append((start_idx, len(states) - 1, current_state, segment_load))
    
    return states, state_means, segments

def compare_with_historical_stages_standalone(current_segments, historical_segments, 
                                              current_feat_df, historical_feat_df,
                                              current_times, historical_times,
                                              current_load, historical_load):
    """
    与历史负荷阶段进行对比分析（独立版本）
    """
    try:
        comparison = {
            'stage_count_comparison': {},
            'aligned_stages': [],
            'significant_differences': [],
            'behavior_explanations': []
        }
        
        # 1. 分析阶段数量变化
        current_count = len(current_segments)
        historical_count = len(historical_segments)
        count_change = current_count - historical_count
        count_change_pct = (count_change / historical_count * 100) if historical_count > 0 else 0
        
        comparison['stage_count_comparison'] = {
            'current_count': current_count,
            'historical_count': historical_count,
            'change': count_change,
            'change_percent': count_change_pct,
            'trend': '增加' if count_change > 0 else ('减少' if count_change < 0 else '不变'),
            'reasons': []
        }
        
        # 解释阶段数量变化的原因
        if count_change > 0:
            comparison['stage_count_comparison']['reasons'].append(
                f'负荷阶段数增加{abs(count_change)}个，可能原因：'
            )
            comparison['stage_count_comparison']['reasons'].append(
                '  1. 用电行为更加多样化，出现更多负荷切换'
            )
            comparison['stage_count_comparison']['reasons'].append(
                '  2. 家庭成员活动模式发生变化'
            )
            comparison['stage_count_comparison']['reasons'].append(
                '  3. 新增用电设备或改变使用习惯'
            )
        elif count_change < 0:
            comparison['stage_count_comparison']['reasons'].append(
                f'负荷阶段数减少{abs(count_change)}个，可能原因：'
            )
            comparison['stage_count_comparison']['reasons'].append(
                '  1. 用电行为更加规律，负荷模式简化'
            )
            comparison['stage_count_comparison']['reasons'].append(
                '  2. 家庭成员减少或外出时间增加'
            )
            comparison['stage_count_comparison']['reasons'].append(
                '  3. 减少了用电设备使用或优化了用电习惯'
            )
        else:
            comparison['stage_count_comparison']['reasons'].append(
                '负荷阶段数保持不变，用电模式相对稳定'
            )
        
        # 2. 逐阶段对齐分析
        for curr_idx, (curr_start, curr_end, curr_state, curr_mean) in enumerate(current_segments):
            curr_start_hour = curr_start * 15 / 60
            curr_end_hour = (curr_end + 1) * 15 / 60
            curr_mid_hour = (curr_start_hour + curr_end_hour) / 2
            
            # 找到历史数据中时间最接近的阶段
            best_match = None
            best_overlap = 0
            best_time_diff = float('inf')
            
            for hist_idx, (hist_start, hist_end, hist_state, hist_mean) in enumerate(historical_segments):
                hist_start_hour = hist_start * 15 / 60
                hist_end_hour = (hist_end + 1) * 15 / 60
                hist_mid_hour = (hist_start_hour + hist_end_hour) / 2
                
                # 计算时间重叠
                overlap_start = max(curr_start_hour, hist_start_hour)
                overlap_end = min(curr_end_hour, hist_end_hour)
                overlap = max(0, overlap_end - overlap_start)
                
                # 计算中心点时间差
                time_diff = abs(curr_mid_hour - hist_mid_hour)
                
                # 选择重叠最大或时间最接近的阶段
                if overlap > best_overlap or (overlap == best_overlap and time_diff < best_time_diff):
                    best_overlap = overlap
                    best_time_diff = time_diff
                    best_match = hist_idx
            
            if best_match is not None:
                hist_start, hist_end, hist_state, hist_mean = historical_segments[best_match]
                
                # 计算负荷差异
                load_diff = curr_mean - hist_mean
                load_diff_pct = (load_diff / hist_mean * 100) if hist_mean != 0 else 0
                
                # 计算时间偏移（阶段的左移或右移）
                hist_start_hour = hist_start * 15 / 60
                hist_end_hour = (hist_end + 1) * 15 / 60
                hist_mid_hour = (hist_start_hour + hist_end_hour) / 2
                
                # 使用阶段的中心点时间来判断整体偏移
                time_shift = curr_mid_hour - hist_mid_hour
                
                aligned_stage = {
                    'current_stage': curr_idx + 1,
                    'historical_stage': best_match + 1,
                    'current_time_range': f"{curr_start_hour:.1f}h-{curr_end_hour:.1f}h",
                    'historical_time_range': f"{hist_start_hour:.1f}h-{hist_end_hour:.1f}h",
                    'current_load': float(curr_mean),
                    'historical_load': float(hist_mean),
                    'load_difference': float(load_diff),
                    'load_difference_percent': float(load_diff_pct),
                    'time_overlap': float(best_overlap),
                    'time_shift': float(time_shift),  # 正值表示右移（推迟），负值表示左移（提前）
                    'environment_diff': {}
                }
                
                comparison['aligned_stages'].append(aligned_stage)
        
        # 3. 识别差异较大的负荷阶段（负荷差异 > 20% 或 时间偏移 >= 1小时）
        for aligned in comparison['aligned_stages']:
            has_load_diff = abs(aligned['load_difference_percent']) > 20
            has_time_shift = abs(aligned['time_shift']) >= 1.0  # 偏移超过1小时
            
            if has_load_diff or has_time_shift:
                diff_info = {
                    'current_stage': aligned['current_stage'],
                    'historical_stage': aligned['historical_stage'],
                    'time_range': aligned['current_time_range'],
                    'historical_time_range': aligned['historical_time_range'],
                    'load_change': aligned['load_difference'],
                    'load_change_percent': aligned['load_difference_percent'],
                    'time_shift': aligned['time_shift'],
                    'change_type': '增加' if aligned['load_difference'] > 0 else '减少',
                    'shift_direction': '右移(推迟)' if aligned['time_shift'] > 0 else ('左移(提前)' if aligned['time_shift'] < 0 else '无偏移'),
                    'explanations': []
                }
                
                # 4. 结合人的行为对差异进行解释
                curr_start_hour = float(aligned['current_time_range'].split('-')[0].replace('h', ''))
                hist_start_hour = float(aligned['historical_time_range'].split('-')[0].replace('h', ''))
                
                # 首先解释时间偏移（阶段的左移或右移）
                if abs(aligned['time_shift']) >= 1.0:
                    shift_hours = abs(aligned['time_shift'])
                    shift_dir = '推迟' if aligned['time_shift'] > 0 else '提前'
                    
                    # 判断阶段类型以提供更具体的解释
                    if 6 <= hist_start_hour < 9 or 6 <= curr_start_hour < 9:
                        diff_info['explanations'].append(
                            f'早高峰阶段时间{shift_dir}约{shift_hours:.1f}小时，可能是：因为周末/假日导致起床时间{shift_dir}、或作息时间调整'
                        )
                    elif 12 <= hist_start_hour < 14 or 12 <= curr_start_hour < 14:
                        diff_info['explanations'].append(
                            f'午间阶段时间{shift_dir}约{shift_hours:.1f}小时，可能是：用餐时间{shift_dir}、或午休习惯改变'
                        )
                    elif 18 <= hist_start_hour < 22 or 18 <= curr_start_hour < 22:
                        diff_info['explanations'].append(
                            f'晚高峰阶段时间{shift_dir}约{shift_hours:.1f}小时，可能是：下班/回家时间{shift_dir}、或晚餐时间调整'
                        )
                    elif 22 <= hist_start_hour or 22 <= curr_start_hour or curr_start_hour < 6 or hist_start_hour < 6:
                        diff_info['explanations'].append(
                            f'夜间阶段时间{shift_dir}约{shift_hours:.1f}小时，可能是：就寝时间{shift_dir}、或夜间活动习惯改变'
                        )
                    else:
                        diff_info['explanations'].append(
                            f'该阶段时间整体{shift_dir}约{shift_hours:.1f}小时，可能是：日常作息时间调整、工作/休息模式改变'
                        )
                
                # 基于时间段和负荷变化的行为解释（仅在负荷差异显著时添加）
                if abs(aligned['load_difference_percent']) > 20:
                    if aligned['load_difference'] > 0:
                        if 6 <= curr_start_hour < 9:
                            diff_info['explanations'].append(
                                '早高峰时段负荷增加，可能是：起床时间提前、早餐准备更复杂、或增加了热水器/咖啡机使用'
                            )
                        elif 9 <= curr_start_hour < 12:
                            diff_info['explanations'].append(
                                '上午时段负荷增加，可能是：在家办公、使用更多电器、或家庭成员未外出'
                            )
                        elif 12 <= curr_start_hour < 14:
                            diff_info['explanations'].append(
                                '午间时段负荷增加，可能是：在家用餐、使用厨房电器增加、或午休期间使用空调/暖气'
                            )
                        elif 14 <= curr_start_hour < 18:
                            diff_info['explanations'].append(
                                '下午时段负荷增加，可能是：在家时间增加、使用娱乐设备、或提前准备晚餐'
                            )
                        elif 18 <= curr_start_hour < 22:
                            diff_info['explanations'].append(
                                '晚高峰时段负荷增加，可能是：回家时间提前、晚餐准备更复杂、家庭娱乐活动增加、或使用更多照明和空调'
                            )
                        else:
                            diff_info['explanations'].append(
                                '夜间时段负荷增加，可能是：就寝时间推迟、夜间使用电器增加、或保持更多设备待机'
                            )
                    else:  # 负荷减少
                        if 6 <= curr_start_hour < 9:
                            diff_info['explanations'].append(
                                '早高峰时段负荷减少，可能是：外出时间提前、简化早餐准备、或减少电器使用'
                            )
                        elif 9 <= curr_start_hour < 12:
                            diff_info['explanations'].append(
                                '上午时段负荷减少，可能是：家庭成员外出增加、减少在家办公、或优化了电器使用'
                            )
                        elif 12 <= curr_start_hour < 14:
                            diff_info['explanations'].append(
                                '午间时段负荷减少，可能是：外出用餐、减少厨房电器使用、或优化了空调使用'
                            )
                        elif 14 <= curr_start_hour < 18:
                            diff_info['explanations'].append(
                                '下午时段负荷减少，可能是：外出时间延长、减少电器待机、或改善了节能习惯'
                            )
                        elif 18 <= curr_start_hour < 22:
                            diff_info['explanations'].append(
                                '晚高峰时段负荷减少，可能是：回家时间推迟、简化晚餐准备、减少娱乐设备使用、或改善照明和空调使用习惯'
                            )
                        else:
                            diff_info['explanations'].append(
                                '夜间时段负荷减少，可能是：就寝时间提前、关闭更多电器、或减少设备待机功耗'
                            )
                
                comparison['significant_differences'].append(diff_info)
        
        # 5. 生成总体行为解释
        if comparison['significant_differences']:
            comparison['behavior_explanations'].append(
                f"共识别出{len(comparison['significant_differences'])}个差异显著的负荷阶段"
            )
            
            increase_count = sum(1 for d in comparison['significant_differences'] if d['load_change'] > 0)
            decrease_count = len(comparison['significant_differences']) - increase_count
            
            # 统计时间偏移模式
            shift_count = sum(1 for d in comparison['significant_differences'] if abs(d.get('time_shift', 0)) >= 1.0)
            right_shift_count = sum(1 for d in comparison['significant_differences'] if d.get('time_shift', 0) >= 1.0)
            left_shift_count = sum(1 for d in comparison['significant_differences'] if d.get('time_shift', 0) <= -1.0)
            
            # 添加时间偏移总体分析
            if shift_count > 0:
                if right_shift_count > left_shift_count:
                    comparison['behavior_explanations'].append(
                        f'时间偏移模式：整体右移(推迟)为主，{shift_count}个阶段有显著时间偏移（{right_shift_count}个右移，{left_shift_count}个左移）'
                    )
                    comparison['behavior_explanations'].append(
                        '偏移原因：可能是周末/假日作息推迟、工作时间调整、或生活习惯改变'
                    )
                elif left_shift_count > right_shift_count:
                    comparison['behavior_explanations'].append(
                        f'时间偏移模式：整体左移(提前)为主，{shift_count}个阶段有显著时间偏移（{left_shift_count}个左移，{right_shift_count}个右移）'
                    )
                    comparison['behavior_explanations'].append(
                        '偏移原因：可能是工作日作息提前、早起习惯养成、或活动时间整体前移'
                    )
                else:
                    comparison['behavior_explanations'].append(
                        f'时间偏移模式：左移和右移并存，{shift_count}个阶段有显著时间偏移'
                    )
                    comparison['behavior_explanations'].append(
                        '偏移原因：不同时段的活动时间调整，用电模式发生重组'
                    )
            
            # 添加负荷变化趋势分析
            if increase_count > decrease_count:
                comparison['behavior_explanations'].append(
                    f'整体趋势：负荷增加为主({increase_count}个阶段增加，{decrease_count}个阶段减少)'
                )
                comparison['behavior_explanations'].append(
                    '可能原因：家庭活动增加、在家时间延长、新增用电设备、或季节性用电需求变化'
                )
            elif decrease_count > increase_count:
                comparison['behavior_explanations'].append(
                    f'整体趋势：负荷减少为主({decrease_count}个阶段减少，{increase_count}个阶段增加)'
                )
                comparison['behavior_explanations'].append(
                    '可能原因：外出时间增加、减少电器使用、节能习惯改善、或季节性用电需求降低'
                )
            else:
                comparison['behavior_explanations'].append(
                    f'整体趋势：增减平衡({increase_count}个阶段增加，{decrease_count}个阶段减少)'
                )
                comparison['behavior_explanations'].append(
                    '可能原因：用电模式调整，不同时段的用电行为发生了变化'
                )
        else:
            comparison['behavior_explanations'].append(
                '各阶段负荷差异较小，用电模式保持相对稳定'
            )
        
        return comparison
        
    except Exception as e:
        print(f"❌ 历史负荷对比分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'stage_count_comparison': {},
            'aligned_stages': [],
            'significant_differences': [],
            'behavior_explanations': [],
            'error': str(e)
        }

def visualize_comparison(current_df, historical_df, current_segments, historical_segments, 
                         comparison, output_path='historical_comparison.png'):
    """可视化历史对比结果"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    hours = np.arange(len(current_df)) * 15 / 60
    
    # 1. 当前负荷曲线与阶段划分
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hours, current_df['load'].values, 'b-', linewidth=2, label='Current Load', alpha=0.7)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(current_segments)))
    for i, (start_idx, end_idx, state, mean_load) in enumerate(current_segments):
        start_hour = start_idx * 15 / 60
        end_hour = (end_idx + 1) * 15 / 60
        ax1.axvspan(start_hour, end_hour, alpha=0.3, color=colors[i])
        ax1.hlines(mean_load, start_hour, end_hour, colors=colors[i], linestyles='--', linewidth=2)
    
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('Load (kW)', fontsize=11)
    ax1.set_title('Current Load with Stage Segmentation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 历史负荷曲线与阶段划分
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(hours, historical_df['load'].values, 'g-', linewidth=2, label='Historical Load', alpha=0.7)
    
    hist_colors = plt.cm.Set3(np.linspace(0, 1, len(historical_segments)))
    for i, (start_idx, end_idx, state, mean_load) in enumerate(historical_segments):
        start_hour = start_idx * 15 / 60
        end_hour = (end_idx + 1) * 15 / 60
        ax2.axvspan(start_hour, end_hour, alpha=0.3, color=hist_colors[i])
        ax2.hlines(mean_load, start_hour, end_hour, colors=hist_colors[i], linestyles='--', linewidth=2)
    
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('Load (kW)', fontsize=11)
    ax2.set_title('Historical Load with Stage Segmentation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 阶段数量对比
    ax3 = fig.add_subplot(gs[1, 0])
    scc = comparison.get('stage_count_comparison', {})
    categories = ['Current', 'Historical']
    counts = [scc.get('current_count', 0), scc.get('historical_count', 0)]
    bars = ax3.bar(categories, counts, color=['blue', 'green'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Stages', fontsize=11)
    ax3.set_title('Stage Count Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. 对齐阶段的负荷对比
    ax4 = fig.add_subplot(gs[1, 1])
    aligned_stages = comparison.get('aligned_stages', [])
    if aligned_stages:
        stage_ids = [s['current_stage'] for s in aligned_stages]
        current_loads = [s['current_load'] for s in aligned_stages]
        historical_loads = [s['historical_load'] for s in aligned_stages]
        
        x = np.arange(len(stage_ids))
        width = 0.35
        
        ax4.bar(x - width/2, current_loads, width, label='Current', color='blue', alpha=0.7)
        ax4.bar(x + width/2, historical_loads, width, label='Historical', color='green', alpha=0.7)
        
        ax4.set_xlabel('Stage ID', fontsize=11)
        ax4.set_ylabel('Average Load (kW)', fontsize=11)
        ax4.set_title('Aligned Stage Load Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(stage_ids)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 负荷差异百分比
    ax5 = fig.add_subplot(gs[2, 0])
    if aligned_stages:
        stage_ids = [s['current_stage'] for s in aligned_stages]
        diff_pcts = [s['load_difference_percent'] for s in aligned_stages]
        
        colors_diff = ['red' if d > 0 else 'green' for d in diff_pcts]
        ax5.bar(stage_ids, diff_pcts, color=colors_diff, alpha=0.7, edgecolor='black')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax5.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='±20% threshold')
        ax5.axhline(y=-20, color='red', linestyle='--', alpha=0.5)
        
        ax5.set_xlabel('Stage ID', fontsize=11)
        ax5.set_ylabel('Load Difference (%)', fontsize=11)
        ax5.set_title('Stage Load Difference Percentage', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 显著差异阶段摘要
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    sig_diffs = comparison.get('significant_differences', [])
    summary_text = f"Significant Differences: {len(sig_diffs)} stages\n" + "="*50 + "\n\n"
    
    for diff in sig_diffs[:5]:  # 只显示前5个
        summary_text += f"Stage {diff['current_stage']} ({diff['time_range']}):\n"
        summary_text += f"  Change: {diff['load_change']:+.3f} kW ({diff['load_change_percent']:+.1f}%)\n"
        if diff['explanations']:
            # 只显示第一条解释（避免文字太多）
            exp = diff['explanations'][0]
            if len(exp) > 60:
                exp = exp[:60] + "..."
            summary_text += f"  Reason: {exp}\n"
        summary_text += "\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. 环境因素对比 - 温度
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(hours, current_df['temperature_current'].values, 'r-', 
             linewidth=2, label='Current Temp', alpha=0.7)
    ax7.plot(hours, historical_df['temperature_current'].values, 'b--', 
             linewidth=2, label='Historical Temp', alpha=0.7)
    ax7.set_xlabel('Time (hours)', fontsize=11)
    ax7.set_ylabel('Temperature (°C)', fontsize=11)
    ax7.set_title('Temperature Comparison', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 行为模式解释
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    
    behavior_exp = comparison.get('behavior_explanations', [])
    behavior_text = "Overall Behavior Analysis\n" + "="*50 + "\n\n"
    for exp in behavior_exp:
        behavior_text += f"• {exp}\n\n"
    
    ax8.text(0.05, 0.95, behavior_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    plt.close()

def main():
    """主演示函数"""
    print("="*80)
    print("Historical Load Stage Comparison Demo")
    print("="*80)
    
    # 1. 生成当前和历史数据
    print("\n1️⃣  Generating demo data...")
    current_df = generate_demo_data('2024-01-15', seed=42, scenario='high_morning')
    historical_df = generate_demo_data('2024-01-08', seed=24, scenario='normal')
    print(f"   Current data: {len(current_df)} time points")
    print(f"   Historical data: {len(historical_df)} time points")
    
    # 2. 负荷阶段划分
    print("\n2️⃣  Segmenting load stages...")
    _, _, current_segments = simple_segmentation(current_df['load'].values, n_segments=4)
    _, _, historical_segments = simple_segmentation(historical_df['load'].values, n_segments=4)
    print(f"   Current stages: {len(current_segments)}")
    print(f"   Historical stages: {len(historical_segments)}")
    
    # 3. 执行历史对比（内嵌函数，避免导入train_household_forecast中的tensorflow依赖）
    print("\n3️⃣  Performing historical comparison...")
    
    comparison = compare_with_historical_stages_standalone(
        current_segments, historical_segments,
        current_df, historical_df,
        current_df.index.tolist(), historical_df.index.tolist(),
        current_df['load'].values, historical_df['load'].values
    )
    
    # 4. 显示对比结果
    print("\n4️⃣  Comparison results:")
    scc = comparison.get('stage_count_comparison', {})
    print(f"\n   Stage count: {scc.get('current_count')} (current) vs {scc.get('historical_count')} (historical)")
    print(f"   Change: {scc.get('change', 0):+d} stages ({scc.get('change_percent', 0):+.1f}%)")
    print(f"   Trend: {scc.get('trend', 'N/A')}")
    
    print(f"\n   Aligned stages: {len(comparison.get('aligned_stages', []))}")
    print(f"   Significant differences: {len(comparison.get('significant_differences', []))}")
    
    # 显示显著差异的阶段
    sig_diffs = comparison.get('significant_differences', [])
    if sig_diffs:
        print(f"\n   Stages with significant differences (>20% change):")
        for diff in sig_diffs[:3]:  # 只显示前3个
            print(f"      Stage {diff['current_stage']} ({diff['time_range']}): "
                  f"{diff['load_change']:+.3f} kW ({diff['load_change_percent']:+.1f}%)")
    
    # 5. 生成可视化
    print("\n5️⃣  Generating visualization...")
    visualize_comparison(current_df, historical_df, current_segments, historical_segments,
                        comparison, output_path='/tmp/historical_comparison_demo.png')
    
    # 6. 生成报告
    print("\n6️⃣  Generating comparison report...")
    
    # 简单的报告生成函数
    def generate_simple_report(comparison, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Historical Load Stage Comparison Report\n")
            f.write("="*80 + "\n\n")
            
            # 阶段数量对比
            scc = comparison.get('stage_count_comparison', {})
            f.write("▶ Stage Count Comparison\n")
            f.write("-"*80 + "\n")
            f.write(f"Current stages: {scc.get('current_count', 0)}\n")
            f.write(f"Historical stages: {scc.get('historical_count', 0)}\n")
            f.write(f"Change: {scc.get('change', 0):+d} stages ({scc.get('change_percent', 0):+.1f}%)\n")
            f.write(f"Trend: {scc.get('trend', 'N/A')}\n\n")
            f.write("Reasons:\n")
            for reason in scc.get('reasons', []):
                f.write(f"{reason}\n")
            
            # 对齐阶段分析
            f.write("\n\n▶ Aligned Stage Analysis\n")
            f.write("-"*80 + "\n")
            for aligned in comparison.get('aligned_stages', [])[:10]:  # 只显示前10个
                f.write(f"\nCurrent Stage {aligned['current_stage']} ↔ Historical Stage {aligned['historical_stage']}:\n")
                f.write(f"  Time: {aligned['current_time_range']} (current) vs {aligned['historical_time_range']} (historical)\n")
                f.write(f"  Load: {aligned['current_load']:.4f} kW (current) vs {aligned['historical_load']:.4f} kW (historical)\n")
                f.write(f"  Difference: {aligned['load_difference']:+.4f} kW ({aligned['load_difference_percent']:+.1f}%)\n")
            
            # 显著差异阶段
            f.write("\n\n▶ Stages with Significant Differences\n")
            f.write("-"*80 + "\n")
            for diff in comparison.get('significant_differences', []):
                f.write(f"\nStage {diff['current_stage']} (Time: {diff['time_range']}):\n")
                f.write(f"  Load change: {diff['load_change']:+.4f} kW ({diff['load_change_percent']:+.1f}%)\n")
                f.write(f"  Type: {diff['change_type']}\n")
                f.write(f"  Explanations:\n")
                for exp in diff['explanations']:
                    f.write(f"    • {exp}\n")
            
            # 总体行为解释
            f.write("\n\n▶ Overall Behavior Analysis\n")
            f.write("-"*80 + "\n")
            for exp in comparison.get('behavior_explanations', []):
                f.write(f"{exp}\n")
    
    report_path = '/tmp/historical_comparison_report.txt'
    generate_simple_report(comparison, report_path)
    print(f"   Report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("✅ Demo completed!")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("  1. Stage count comparison - Analyze increase/decrease in stage numbers")
    print("  2. Stage alignment - Match stages between current and historical data")
    print("  3. Difference detection - Identify stages with significant load changes")
    print("  4. Behavior explanations - Interpret changes based on human behavior patterns")
    print("  5. Environmental factors - Consider temperature, humidity in analysis")
    print("\nOutput Files:")
    print(f"  • Visualization: /tmp/historical_comparison_demo.png")
    print(f"  • Report: /tmp/historical_comparison_report.txt")

if __name__ == '__main__':
    main()
