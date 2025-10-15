#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
负荷预测可解释性功能示例
=====================================

本脚本展示如何使用系统中已实现的可解释性功能来分析负荷预测结果。

运行方式：
    python example_interpretability.py

功能：
    1. 生成模拟的负荷预测数据
    2. 进行智能阶段划分
    3. 分析各阶段特征和影响因素
    4. 生成可解释性报告和可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# 设置matplotlib以支持中文（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def generate_sample_load_data():
    """
    生成示例负荷数据（模拟一天24小时的家庭用电情况）
    
    Returns:
        pd.DataFrame: 包含时间、负荷值和环境特征的数据框
    """
    print("📊 生成示例负荷数据...")
    
    # 生成24小时，每15分钟一个点，共96个点
    n_points = 96
    times = pd.date_range('2024-01-01 00:00', periods=n_points, freq='15min')
    hours = np.arange(n_points) / 4  # 转换为小时
    
    # 模拟典型家庭用电模式
    load = np.zeros(n_points)
    for i, h in enumerate(hours):
        if h < 6:  # 夜间 (0-6时): 低负荷
            load[i] = 0.5 + np.random.normal(0, 0.05)
        elif h < 9:  # 早高峰 (6-9时): 负荷上升
            progress = (h - 6) / 3
            load[i] = 0.5 + progress * 2.0 + np.random.normal(0, 0.1)
        elif h < 18:  # 白天 (9-18时): 中等负荷
            load[i] = 1.0 + np.random.normal(0, 0.1)
        elif h < 22:  # 晚高峰 (18-22时): 高负荷
            progress = (h - 18) / 4
            load[i] = 1.0 + progress * 2.5 + np.random.normal(0, 0.15)
        else:  # 深夜 (22-24时): 负荷下降
            progress = (h - 22) / 2
            load[i] = 3.5 - progress * 2.7 + np.random.normal(0, 0.1)
    
    load = np.maximum(load, 0.3)  # 确保负荷为正
    
    # 生成相关的环境特征
    temperature = 15 + 10 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 1, n_points)
    humidity = 60 + 15 * np.cos((hours - 3) * np.pi / 12) + np.random.normal(0, 2, n_points)
    cloudCover = np.clip(0.3 + 0.4 * np.sin(hours * np.pi / 24) + np.random.normal(0, 0.1, n_points), 0, 1)
    
    df = pd.DataFrame({
        'time': times,
        'load': load,
        'temperature_current': temperature,
        'humidity_current': humidity,
        'cloudCover_current': cloudCover,
        'hour': hours
    })
    
    df = df.set_index('time')
    
    print(f"✅ 生成了 {len(df)} 个时间点的数据")
    print(f"   负荷范围: {load.min():.2f} - {load.max():.2f} kW")
    print(f"   温度范围: {temperature.min():.1f} - {temperature.max():.1f} °C")
    
    return df

def segment_load_by_threshold(load_values, n_segments=4, min_segment_length=6):
    """
    基于负荷水平进行改进的智能分段
    
    使用时间序列平滑、变化点检测和最小段长度约束来创建更精准的阶段划分。
    增强版：使用时间特征的正余弦编码让时间成为连续的
    
    Args:
        load_values: 负荷值数组
        n_segments: 目标段数（建议范围：3-8）
        min_segment_length: 最小段长度（时间点数），默认6个点（1.5小时）
        
    Returns:
        list: 段落信息 [(start_idx, end_idx, state, mean_load), ...]
    """
    load_values = np.array(load_values)
    n = len(load_values)
    
    # 构建时间特征（正余弦编码让时间成为连续的）
    time_features = []
    for i in range(n):
        hour = (i * 0.25) % 24  # 假设15分钟间隔
        time_features.append([
            np.sin(2 * np.pi * hour / 24),  # 小时的正弦编码
            np.cos(2 * np.pi * hour / 24),  # 小时的余弦编码
            np.sin(2 * np.pi * (i % 96) / 96),  # 日内位置编码
            np.cos(2 * np.pi * (i % 96) / 96)
        ])
    time_features = np.array(time_features)
    
    # 步骤1: 对负荷值进行平滑处理，减少噪声影响
    # 使用移动平均窗口大小为5（约1小时）
    window_size = 5
    smoothed_load = np.convolve(load_values, np.ones(window_size)/window_size, mode='same')
    
    # 步骤2: 计算一阶导数（变化率）来识别显著变化点
    # 使用中心差分法计算导数
    derivative = np.zeros_like(smoothed_load)
    derivative[1:-1] = (smoothed_load[2:] - smoothed_load[:-2]) / 2
    derivative[0] = smoothed_load[1] - smoothed_load[0]
    derivative[-1] = smoothed_load[-1] - smoothed_load[-2]
    
    # 步骤3: 识别显著的变化点
    # 计算导数的标准差作为阈值
    derivative_std = np.std(derivative)
    derivative_threshold = derivative_std * 0.5  # 使用0.5倍标准差作为阈值
    
    # 找出导数绝对值较大的点（显著变化点）
    change_points = []
    for i in range(1, n - 1):
        # 如果导数符号改变或导数绝对值超过阈值，标记为变化点
        if (abs(derivative[i]) > derivative_threshold and 
            (derivative[i] * derivative[i-1] < 0 or abs(derivative[i]) > abs(derivative[i-1]) * 1.5)):
            change_points.append(i)
    
    # 归一化负荷值以便与时间特征结合
    load_normalized = (smoothed_load - smoothed_load.min()) / (smoothed_load.max() - smoothed_load.min() + 1e-10)
    
    # 步骤4: 基于变化点和负荷水平进行聚类分段（结合时间特征）
    # 使用K-means思想对平滑后的负荷进行聚类
    thresholds = np.percentile(smoothed_load, np.linspace(0, 100, n_segments + 1))
    states = np.digitize(smoothed_load, thresholds[1:-1])
    
    # 使用时间特征优化状态边界
    for i in range(1, n - 1):
        if states[i] != states[i-1]:
            # 计算时间相似度
            time_sim_prev = np.dot(time_features[i], time_features[i-1])
            
            # 如果时间特征变化不显著，且负荷差异小，则合并状态
            if time_sim_prev > 0.95 and abs(load_normalized[i] - load_normalized[i-1]) < 0.1:
                states[i] = states[i-1]
    
    # 步骤5: 识别初始段
    initial_segments = []
    current_state = states[0]
    start_idx = 0
    
    for i in range(1, n):
        # 如果状态改变或遇到显著变化点，结束当前段
        if states[i] != current_state or i in change_points:
            if i - start_idx >= 2:  # 至少2个点才形成段
                segment_load = load_values[start_idx:i]
                initial_segments.append((start_idx, i-1, current_state, np.mean(segment_load)))
                start_idx = i
                current_state = states[i]
    
    # 添加最后一段
    if start_idx < n:
        segment_load = load_values[start_idx:]
        initial_segments.append((start_idx, n-1, current_state, np.mean(segment_load)))
    
    # 步骤6: 合并短小和相似的段
    merged_segments = []
    i = 0
    
    while i < len(initial_segments):
        start_idx, end_idx, state, mean_load = initial_segments[i]
        segment_length = end_idx - start_idx + 1
        
        # 如果当前段太短，尝试与相邻段合并
        if segment_length < min_segment_length and len(merged_segments) > 0:
            # 与前一个段合并
            prev_start, prev_end, prev_state, prev_mean = merged_segments[-1]
            combined_load = np.mean(load_values[prev_start:end_idx+1])
            
            # 选择负荷水平更接近的状态
            if abs(combined_load - thresholds[prev_state]) <= abs(combined_load - thresholds[state]):
                final_state = prev_state
            else:
                final_state = state
            
            merged_segments[-1] = (prev_start, end_idx, final_state, combined_load)
        else:
            # 如果段足够长，或者是第一个段，直接添加
            merged_segments.append((start_idx, end_idx, state, mean_load))
        
        i += 1
    
    # 步骤7: 进一步合并负荷水平相似的相邻段
    final_segments = []
    i = 0
    
    while i < len(merged_segments):
        start_idx, end_idx, state, mean_load = merged_segments[i]
        
        # 检查是否可以与下一个段合并
        if i < len(merged_segments) - 1:
            next_start, next_end, next_state, next_mean = merged_segments[i + 1]
            
            # 如果相邻两段的负荷水平差异小于15%，合并它们
            load_diff_pct = abs(next_mean - mean_load) / mean_load * 100 if mean_load > 0 else 0
            
            if load_diff_pct < 15:
                # 合并两段
                combined_load = np.mean(load_values[start_idx:next_end+1])
                # 选择负荷水平更高的状态（保持语义一致性）
                final_state = state if mean_load >= next_mean else next_state
                final_segments.append((start_idx, next_end, final_state, combined_load))
                i += 2  # 跳过下一个段
                continue
        
        final_segments.append((start_idx, end_idx, state, mean_load))
        i += 1
    
    return final_segments

def analyze_segments(segments, df):
    """
    分析各阶段的特征和影响因素
    
    Args:
        segments: 段落信息列表
        df: 特征数据框
        
    Returns:
        list: 各阶段的详细分析结果
    """
    print("\n🔍 分析各阶段特征...")
    
    segment_analysis = []
    all_means = [seg[3] for seg in segments]
    
    for i, (start_idx, end_idx, state, mean_load) in enumerate(segments):
        # 时间信息
        start_time = start_idx * 0.25  # 15分钟 = 0.25小时
        end_time = (end_idx + 1) * 0.25
        duration = end_time - start_time
        
        # 负荷水平分类
        load_percentile = (sorted(all_means).index(mean_load) + 1) / len(all_means)
        if load_percentile <= 0.25:
            load_level = '低负荷'
        elif load_percentile <= 0.5:
            load_level = '中低负荷'
        elif load_percentile <= 0.75:
            load_level = '中高负荷'
        else:
            load_level = '高负荷'
        
        # 提取该段的特征
        segment_df = df.iloc[start_idx:end_idx+1]
        avg_temp = segment_df['temperature_current'].mean()
        avg_humidity = segment_df['humidity_current'].mean()
        avg_cloud = segment_df['cloudCover_current'].mean()
        avg_hour = segment_df['hour'].mean()
        
        # 识别关键影响因素
        key_factors = []
        
        # 温度影响
        if avg_temp > 25:
            key_factors.append(f'高温({avg_temp:.1f}°C)可能增加空调负荷')
        elif avg_temp < 10:
            key_factors.append(f'低温({avg_temp:.1f}°C)可能增加供暖负荷')
        else:
            key_factors.append(f'温度适中({avg_temp:.1f}°C)')
        
        # 时间段特征
        if 6 <= avg_hour < 9:
            key_factors.append('早高峰 - 起床、早餐活动')
        elif 9 <= avg_hour < 18:
            key_factors.append('白天 - 多数家庭成员外出，基础负荷')
        elif 18 <= avg_hour < 22:
            key_factors.append('晚高峰 - 回家、晚餐、娱乐活动')
        else:
            key_factors.append('夜间 - 睡眠、待机负荷')
        
        # 湿度影响
        if avg_humidity > 70:
            key_factors.append(f'高湿度({avg_humidity:.0f}%)可能增加除湿需求')
        
        # 云量影响
        if avg_cloud > 0.7:
            key_factors.append(f'多云({avg_cloud:.1f})减少自然采光，可能增加照明负荷')
        
        segment_info = {
            'segment_id': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration_hours': duration,
            'state': state,
            'mean_load': mean_load,
            'load_level': load_level,
            'key_factors': key_factors,
            'avg_temp': avg_temp,
            'avg_humidity': avg_humidity
        }
        
        segment_analysis.append(segment_info)
        
        print(f"  阶段{i+1}: {start_time:.1f}h-{end_time:.1f}h, {load_level} ({mean_load:.2f} kW)")
        print(f"    关键因素: {key_factors[0]}")
    
    return segment_analysis

def generate_report(segment_analysis, output_path):
    """
    生成可解释性分析报告（文本格式）
    
    Args:
        segment_analysis: 阶段分析结果
        output_path: 报告保存路径
    """
    print(f"\n📝 生成可解释性报告...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("负荷预测可解释性分析报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析阶段数: {len(segment_analysis)}\n\n")
        
        f.write("【阶段详细分析】\n")
        f.write("-" * 80 + "\n\n")
        
        for seg in segment_analysis:
            f.write(f"阶段 {seg['segment_id']}:\n")
            f.write(f"  时间范围: {seg['start_time']:.1f}h - {seg['end_time']:.1f}h ")
            f.write(f"(持续 {seg['duration_hours']:.1f} 小时)\n")
            f.write(f"  负荷水平: {seg['load_level']}\n")
            f.write(f"  平均负荷: {seg['mean_load']:.3f} kW\n")
            f.write(f"  平均温度: {seg['avg_temp']:.1f} °C\n")
            f.write(f"  平均湿度: {seg['avg_humidity']:.0f} %\n")
            f.write(f"  关键影响因素:\n")
            for factor in seg['key_factors']:
                f.write(f"    • {factor}\n")
            f.write("\n")
        
        # 趋势变化分析
        if len(segment_analysis) > 1:
            f.write("【阶段间趋势变化】\n")
            f.write("-" * 80 + "\n\n")
            
            for i in range(len(segment_analysis) - 1):
                curr = segment_analysis[i]
                next_seg = segment_analysis[i + 1]
                
                load_change = next_seg['mean_load'] - curr['mean_load']
                load_change_pct = (load_change / curr['mean_load'] * 100) if curr['mean_load'] != 0 else 0
                
                if abs(load_change_pct) < 10:
                    trend = "基本稳定"
                elif load_change_pct > 50:
                    trend = "显著上升"
                elif load_change_pct > 0:
                    trend = "上升"
                elif load_change_pct < -50:
                    trend = "显著下降"
                else:
                    trend = "下降"
                
                f.write(f"阶段 {curr['segment_id']} → 阶段 {next_seg['segment_id']}:\n")
                f.write(f"  变化趋势: {trend}\n")
                f.write(f"  负荷变化: {load_change:+.3f} kW ({load_change_pct:+.1f}%)\n")
                f.write(f"  原因分析:\n")
                f.write(f"    • 负荷从 {curr['load_level']} 变为 {next_seg['load_level']}\n")
                f.write(f"    • 时段从 {curr['start_time']:.0f}h 转换到 {next_seg['start_time']:.0f}h\n")
                f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
    
    print(f"✅ 报告已保存到: {output_path}")

def create_visualization(df, segments, segment_analysis, output_path):
    """
    创建可解释性分析可视化图表
    
    Args:
        df: 数据框
        segments: 段落信息
        segment_analysis: 阶段分析结果
        output_path: 图片保存路径
    """
    print(f"\n📊 生成可视化图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('负荷预测可解释性分析', fontsize=16, fontweight='bold')
    
    # 1. 负荷曲线与阶段划分
    ax1 = axes[0, 0]
    hours = df['hour'].values
    load = df['load'].values
    
    # 为不同阶段绘制不同颜色的背景
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
    for i, (start_idx, end_idx, state, mean_load) in enumerate(segments):
        start_hour = start_idx * 0.25
        end_hour = (end_idx + 1) * 0.25
        ax1.axvspan(start_hour, end_hour, alpha=0.3, color=colors[i])
        ax1.hlines(mean_load, start_hour, end_hour, colors=colors[i], 
                  linestyles='--', linewidth=2, alpha=0.8)
    
    ax1.plot(hours, load, 'b-', linewidth=2, label='负荷曲线')
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('负荷 (kW)')
    ax1.set_title('负荷曲线与阶段划分')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 各阶段负荷水平对比
    ax2 = axes[0, 1]
    seg_ids = [seg['segment_id'] for seg in segment_analysis]
    seg_loads = [seg['mean_load'] for seg in segment_analysis]
    seg_colors = [colors[i] for i in range(len(segment_analysis))]
    
    bars = ax2.bar(seg_ids, seg_loads, color=seg_colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('阶段编号')
    ax2.set_ylabel('平均负荷 (kW)')
    ax2.set_title('各阶段平均负荷对比')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, load in zip(bars, seg_loads):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{load:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 环境因素变化
    ax3 = axes[1, 0]
    ax3_temp = ax3.twinx()
    
    ax3.plot(hours, df['temperature_current'], 'r-', linewidth=2, label='温度', alpha=0.7)
    ax3_temp.plot(hours, df['humidity_current'], 'b-', linewidth=2, label='湿度', alpha=0.7)
    
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('温度 (°C)', color='r')
    ax3_temp.set_ylabel('湿度 (%)', color='b')
    ax3.set_title('环境因素变化趋势')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3_temp.tick_params(axis='y', labelcolor='b')
    ax3.grid(True, alpha=0.3)
    
    # 添加图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_temp.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 4. 负荷水平分布
    ax4 = axes[1, 1]
    level_counts = {}
    for seg in segment_analysis:
        level = seg['load_level']
        level_counts[level] = level_counts.get(level, 0) + seg['duration_hours']
    
    levels = list(level_counts.keys())
    durations = list(level_counts.values())
    level_colors = {'低负荷': '#90EE90', '中低负荷': '#FFD700', 
                   '中高负荷': '#FFA500', '高负荷': '#FF6347'}
    bar_colors = [level_colors.get(l, 'gray') for l in levels]
    
    ax4.bar(levels, durations, color=bar_colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('负荷水平')
    ax4.set_ylabel('持续时间 (小时)')
    ax4.set_title('负荷水平时间分布')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (level, duration) in enumerate(zip(levels, durations)):
        ax4.text(i, duration, f'{duration:.1f}h', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")
    plt.close()

def main():
    """
    主函数：演示完整的可解释性分析流程
    """
    print("\n" + "=" * 80)
    print("负荷预测可解释性功能演示")
    print("=" * 80)
    print("\n本演示展示系统中已实现的可解释性功能\n")
    
    # 1. 生成示例数据
    df = generate_sample_load_data()
    
    # 2. 负荷阶段划分
    print("\n🔄 进行智能负荷阶段划分...")
    segments = segment_load_by_threshold(df['load'].values, n_segments=5)
    print(f"✅ 识别出 {len(segments)} 个负荷阶段")
    
    # 3. 分析各阶段
    segment_analysis = analyze_segments(segments, df)
    
    # 4. 生成报告
    output_dir = '/tmp'
    report_path = os.path.join(output_dir, 'interpretability_example_report.txt')
    generate_report(segment_analysis, report_path)
    
    # 5. 生成可视化
    viz_path = os.path.join(output_dir, 'interpretability_example_viz.png')
    create_visualization(df, segments, segment_analysis, viz_path)
    
    # 6. 打印摘要
    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    print("\n📁 生成的文件：")
    print(f"  - 分析报告: {report_path}")
    print(f"  - 可视化图: {viz_path}")
    
    print("\n💡 主要功能：")
    print("  1. ✓ 智能负荷阶段划分 - 自动识别不同的用电阶段")
    print("  2. ✓ 阶段特征分析 - 分析每个阶段的负荷水平和特点")
    print("  3. ✓ 影响因素识别 - 识别温度、湿度、时间等关键因素")
    print("  4. ✓ 趋势变化解释 - 解释阶段间负荷变化的原因")
    print("  5. ✓ 自动报告生成 - 生成详细的文本和图形报告")
    
    print("\n📖 这些功能已完全集成到预测系统中！")
    print("   运行 'python train_household_forecast.py' 进行预测时会自动生成。")
    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    main()
