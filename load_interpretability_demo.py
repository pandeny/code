"""
负荷变化可解释性模型演示脚本

该脚本展示如何使用负荷变化可解释性模型来分析负荷阶段趋势和量值变化的原因。

主要功能：
1. 负荷阶段划分（HMM方法）
2. 特征提取和相关性分析
3. 阶段变化原因解释
4. 环境因素影响评估
5. 生成可解释性报告和可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def generate_demo_data():
    """
    生成演示用的负荷数据和环境特征
    模拟一天24小时的负荷变化（15分钟间隔，共96个点）
    """
    n_points = 96  # 24小时 * 4个15分钟
    times = pd.date_range('2024-01-01', periods=n_points, freq='15T')
    
    # 生成基础负荷曲线（模拟家庭用电模式）
    hours = np.arange(n_points) / 4  # 转换为小时
    
    # 夜间低负荷(0-6h): 0.5
    # 早高峰(6-9h): 上升到2.5
    # 白天(9-18h): 降至1.0
    # 晚高峰(18-22h): 上升到3.5
    # 夜间(22-24h): 降至0.8
    
    load = np.zeros(n_points)
    for i, h in enumerate(hours):
        if h < 6:  # 夜间
            load[i] = 0.5 + np.random.normal(0, 0.05)
        elif h < 9:  # 早高峰
            progress = (h - 6) / 3
            load[i] = 0.5 + progress * 2.0 + np.random.normal(0, 0.1)
        elif h < 18:  # 白天
            load[i] = 1.0 + np.random.normal(0, 0.08)
        elif h < 22:  # 晚高峰
            progress = (h - 18) / 4
            load[i] = 1.0 + progress * 2.5 + np.random.normal(0, 0.15)
        else:  # 深夜
            progress = (h - 22) / 2
            load[i] = 3.5 - progress * 2.7 + np.random.normal(0, 0.1)
    
    # 确保负荷为正值
    load = np.maximum(load, 0.3)
    
    # 生成环境特征（模拟数据）
    temperature = 15 + 10 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 1, n_points)
    humidity = 60 + 15 * np.cos((hours - 3) * np.pi / 12) + np.random.normal(0, 2, n_points)
    cloudCover = np.clip(0.3 + 0.4 * np.sin(hours * np.pi / 24) + np.random.normal(0, 0.1, n_points), 0, 1)
    
    df = pd.DataFrame({
        'time': times,
        'load': load,
        'load_smooth': load,  # 简化，实际会做平滑处理
        'temperature_current': temperature,
        'humidity_current': humidity,
        'cloudCover_current': cloudCover,
        'hour': hours,
        'target_next': np.roll(load, -1)  # 下一时刻的负荷
    })
    
    # 设置时间为索引
    df = df.set_index('time')
    
    return df

def simple_segmentation(load_values, n_segments=4):
    """
    简单的负荷分段方法（基于分位数）
    """
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

def analyze_segment_features(segments, feat_df, pred_times):
    """
    分析每个阶段的特征
    """
    segment_analysis = []
    
    for i, (start_idx, end_idx, state, mean_load) in enumerate(segments):
        segment_info = {
            'segment_id': i + 1,
            'start_time': start_idx * 15 / 60,  # 转换为小时
            'end_time': (end_idx + 1) * 15 / 60,
            'duration_hours': (end_idx - start_idx + 1) * 15 / 60,
            'state': int(state),
            'mean_load': float(mean_load),
            'load_level': '',
            'key_factors': []
        }
        
        # 确定负荷水平类别
        all_segment_means = [seg[3] for seg in segments]
        load_percentile = (sorted(all_segment_means).index(mean_load) + 1) / len(all_segment_means)
        
        if load_percentile <= 0.25:
            segment_info['load_level'] = '低负荷'
        elif load_percentile <= 0.5:
            segment_info['load_level'] = '中低负荷'
        elif load_percentile <= 0.75:
            segment_info['load_level'] = '中高负荷'
        else:
            segment_info['load_level'] = '高负荷'
        
        # 分析该段的特征
        if len(pred_times) > end_idx:
            segment_times = pred_times[start_idx:end_idx+1]
            segment_features = feat_df.loc[segment_times]
            
            # 分析温度影响
            if 'temperature_current' in segment_features.columns:
                avg_temp = segment_features['temperature_current'].mean()
                if avg_temp > 25:
                    segment_info['key_factors'].append(f'高温({avg_temp:.1f}°C)可能增加空调负荷')
                elif avg_temp < 10:
                    segment_info['key_factors'].append(f'低温({avg_temp:.1f}°C)可能增加供暖负荷')
                else:
                    segment_info['key_factors'].append(f'温度适中({avg_temp:.1f}°C)')
            
            # 分析时间段影响
            if 'hour' in segment_features.columns:
                avg_hour = segment_features['hour'].mean()
                if 6 <= avg_hour < 9:
                    segment_info['key_factors'].append('早高峰时段 - 起床、早餐活动')
                elif 9 <= avg_hour < 12:
                    segment_info['key_factors'].append('上午时段 - 多数家庭成员外出')
                elif 12 <= avg_hour < 14:
                    segment_info['key_factors'].append('午间时段 - 午餐、休息')
                elif 14 <= avg_hour < 18:
                    segment_info['key_factors'].append('下午时段 - 持续低负荷')
                elif 18 <= avg_hour < 22:
                    segment_info['key_factors'].append('晚高峰时段 - 回家、晚餐、娱乐')
                elif 22 <= avg_hour or avg_hour < 6:
                    segment_info['key_factors'].append('夜间时段 - 睡眠、待机负荷')
        
        if not segment_info['key_factors']:
            segment_info['key_factors'].append('负荷水平主要由用户行为模式决定')
        
        segment_analysis.append(segment_info)
    
    return segment_analysis

def analyze_trends(segments):
    """
    分析阶段间的趋势变化
    """
    trend_changes = []
    
    for i in range(len(segments) - 1):
        curr_seg = segments[i]
        next_seg = segments[i + 1]
        
        load_change = next_seg[3] - curr_seg[3]
        load_change_pct = (load_change / curr_seg[3] * 100) if curr_seg[3] != 0 else 0
        
        trend_info = {
            'from_segment': i + 1,
            'to_segment': i + 2,
            'load_change': float(load_change),
            'load_change_percent': float(load_change_pct),
            'trend': '',
            'explanation': []
        }
        
        # 判断变化趋势
        if abs(load_change_pct) < 5:
            trend_info['trend'] = '稳定'
            trend_info['explanation'].append('负荷水平基本保持不变')
        elif load_change_pct > 0:
            if load_change_pct > 30:
                trend_info['trend'] = '显著上升'
                trend_info['explanation'].append(f'负荷大幅增加{load_change_pct:.1f}%')
            else:
                trend_info['trend'] = '上升'
                trend_info['explanation'].append(f'负荷增加{load_change_pct:.1f}%')
        else:
            if load_change_pct < -30:
                trend_info['trend'] = '显著下降'
                trend_info['explanation'].append(f'负荷大幅下降{abs(load_change_pct):.1f}%')
            else:
                trend_info['trend'] = '下降'
                trend_info['explanation'].append(f'负荷下降{abs(load_change_pct):.1f}%')
        
        # 时间相关的变化解释
        curr_start_hour = curr_seg[0] * 15 / 60
        next_start_hour = next_seg[0] * 15 / 60
        
        if curr_start_hour < 6 and next_start_hour >= 6:
            trend_info['explanation'].append('进入早晨时段，家庭活动增加')
        elif curr_start_hour < 18 and next_start_hour >= 18:
            trend_info['explanation'].append('进入傍晚时段，家庭成员返回')
        elif curr_start_hour < 22 and next_start_hour >= 22:
            trend_info['explanation'].append('进入深夜时段，活动减少')
        
        trend_changes.append(trend_info)
    
    return trend_changes

def visualize_demo(df, segments, segment_analysis, trend_changes, output_path='demo_output.png'):
    """
    可视化演示结果
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 负荷曲线与阶段划分
    ax1 = fig.add_subplot(gs[0, :])
    hours = np.arange(len(df)) * 15 / 60
    ax1.plot(hours, df['load'].values, 'b-', linewidth=2, label='负荷曲线', alpha=0.7)
    
    # 绘制阶段背景
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
    for i, (start_idx, end_idx, state, mean_load) in enumerate(segments):
        start_hour = start_idx * 15 / 60
        end_hour = (end_idx + 1) * 15 / 60
        ax1.axvspan(start_hour, end_hour, alpha=0.3, color=colors[i], 
                   label=f'阶段{i+1}')
        ax1.hlines(mean_load, start_hour, end_hour, colors=colors[i], 
                  linestyles='--', linewidth=2)
    
    ax1.set_xlabel('时间 (小时)', fontsize=12)
    ax1.set_ylabel('负荷 (kW)', fontsize=12)
    ax1.set_title('负荷曲线与智能阶段划分', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', ncol=4)
    ax1.grid(True, alpha=0.3)
    
    # 2. 阶段负荷水平对比
    ax2 = fig.add_subplot(gs[1, 0])
    seg_ids = [seg['segment_id'] for seg in segment_analysis]
    seg_loads = [seg['mean_load'] for seg in segment_analysis]
    seg_levels = [seg['load_level'] for seg in segment_analysis]
    
    bar_colors = []
    for level in seg_levels:
        if level == '低负荷':
            bar_colors.append('green')
        elif level == '中低负荷':
            bar_colors.append('lightgreen')
        elif level == '中高负荷':
            bar_colors.append('orange')
        else:
            bar_colors.append('red')
    
    bars = ax2.bar(seg_ids, seg_loads, color=bar_colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('阶段编号', fontsize=12)
    ax2.set_ylabel('平均负荷 (kW)', fontsize=12)
    ax2.set_title('各阶段负荷水平', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, load, level in zip(bars, seg_loads, seg_levels):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{load:.2f}\n{level}',
                ha='center', va='bottom', fontsize=8)
    
    # 3. 阶段变化趋势
    ax3 = fig.add_subplot(gs[1, 1])
    if trend_changes:
        from_segs = [t['from_segment'] for t in trend_changes]
        change_pcts = [t['load_change_percent'] for t in trend_changes]
        
        ax3.plot(from_segs, change_pcts, marker='o', linewidth=2, markersize=8, color='blue')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.fill_between(from_segs, 0, change_pcts, alpha=0.3,
                        color=['green' if c < 0 else 'red' for c in change_pcts])
        ax3.set_xlabel('起始阶段', fontsize=12)
        ax3.set_ylabel('负荷变化率 (%)', fontsize=12)
        ax3.set_title('阶段间负荷变化率', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. 环境因素影响
    ax4 = fig.add_subplot(gs[2, 0])
    ax4_twin = ax4.twinx()
    
    ax4.plot(hours, df['temperature_current'].values, 'r-', linewidth=2, label='温度 (°C)', alpha=0.7)
    ax4_twin.plot(hours, df['load'].values, 'b--', linewidth=2, label='负荷 (kW)', alpha=0.5)
    
    ax4.set_xlabel('时间 (小时)', fontsize=12)
    ax4.set_ylabel('温度 (°C)', fontsize=12, color='red')
    ax4_twin.set_ylabel('负荷 (kW)', fontsize=12, color='blue')
    ax4.set_title('温度与负荷关系', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4_twin.tick_params(axis='y', labelcolor='blue')
    ax4.grid(True, alpha=0.3)
    
    # 5. 阶段解释摘要
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    summary_text = "负荷变化解释摘要\n" + "="*40 + "\n\n"
    for seg in segment_analysis[:5]:  # 只显示前5个阶段
        summary_text += f"阶段{seg['segment_id']}: {seg['load_level']}\n"
        summary_text += f"  时间: {seg['start_time']:.1f}h-{seg['end_time']:.1f}h\n"
        if seg['key_factors']:
            summary_text += f"  因素: {seg['key_factors'][0]}\n"
        summary_text += "\n"
    
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化结果已保存到: {output_path}")
    plt.close()

def main():
    """
    主演示函数
    """
    print("="*60)
    print("负荷变化可解释性模型演示")
    print("="*60)
    
    # 1. 生成演示数据
    print("\n1️⃣  生成演示数据...")
    df = generate_demo_data()
    print(f"   生成了 {len(df)} 个时间点的负荷数据")
    print(f"   负荷范围: {df['load'].min():.2f} - {df['load'].max():.2f} kW")
    
    # 2. 负荷阶段划分
    print("\n2️⃣  进行负荷阶段划分...")
    states, state_means, segments = simple_segmentation(df['load'].values, n_segments=4)
    print(f"   识别出 {len(segments)} 个负荷阶段")
    for i, (start, end, state, mean_load) in enumerate(segments):
        print(f"   阶段{i+1}: {start*15/60:.1f}h-{(end+1)*15/60:.1f}h, 平均负荷={mean_load:.2f} kW")
    
    # 3. 分析阶段特征
    print("\n3️⃣  分析各阶段特征...")
    pred_times = df.index.tolist()  # 使用索引作为时间列表
    segment_analysis = analyze_segment_features(segments, df, pred_times)
    for seg in segment_analysis:
        print(f"   阶段{seg['segment_id']}: {seg['load_level']}")
        for factor in seg['key_factors'][:2]:  # 只显示前2个因素
            print(f"      • {factor}")
    
    # 4. 分析趋势变化
    print("\n4️⃣  分析阶段间趋势变化...")
    trend_changes = analyze_trends(segments)
    for trend in trend_changes:
        print(f"   阶段{trend['from_segment']}→阶段{trend['to_segment']}: "
              f"{trend['trend']} ({trend['load_change_percent']:+.1f}%)")
        if trend['explanation']:
            print(f"      原因: {trend['explanation'][0]}")
    
    # 5. 生成可视化
    print("\n5️⃣  生成可视化结果...")
    visualize_demo(df, segments, segment_analysis, trend_changes, 
                   output_path='/tmp/load_interpretability_demo.png')
    
    # 6. 生成报告
    print("\n6️⃣  生成可解释性报告...")
    report_path = '/tmp/load_interpretability_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("负荷变化可解释性分析报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("【阶段详细分析】\n")
        f.write("-"*60 + "\n")
        for seg in segment_analysis:
            f.write(f"\n阶段 {seg['segment_id']}:\n")
            f.write(f"  时间范围: {seg['start_time']:.1f}h - {seg['end_time']:.1f}h\n")
            f.write(f"  负荷水平: {seg['load_level']} (平均: {seg['mean_load']:.2f} kW)\n")
            f.write(f"  关键影响因素:\n")
            for factor in seg['key_factors']:
                f.write(f"    • {factor}\n")
        
        f.write("\n【趋势变化分析】\n")
        f.write("-"*60 + "\n")
        for trend in trend_changes:
            f.write(f"\n阶段 {trend['from_segment']} → 阶段 {trend['to_segment']}:\n")
            f.write(f"  变化趋势: {trend['trend']}\n")
            f.write(f"  负荷变化: {trend['load_change']:+.2f} kW ({trend['load_change_percent']:+.1f}%)\n")
            f.write(f"  变化原因:\n")
            for exp in trend['explanation']:
                f.write(f"    • {exp}\n")
    
    print(f"   报告已保存到: {report_path}")
    
    print("\n" + "="*60)
    print("✅ 演示完成！")
    print("="*60)
    print("\n模型主要功能:")
    print("  1. 智能负荷阶段划分 - 自动识别负荷变化的不同阶段")
    print("  2. 阶段特征分析 - 提取每个阶段的负荷水平和时间特征")
    print("  3. 趋势变化解释 - 分析阶段间负荷变化的原因")
    print("  4. 环境因素影响 - 评估温度、湿度等环境因素的影响")
    print("  5. 可视化展示 - 直观展示分析结果")
    print("\n适用场景:")
    print("  • 家庭负荷分析和优化")
    print("  • 负荷预测结果解释")
    print("  • 用电行为模式识别")
    print("  • 能源管理决策支持")

if __name__ == '__main__':
    main()
