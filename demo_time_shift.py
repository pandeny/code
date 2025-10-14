"""
示例：展示预测日负荷阶段时间偏移检测功能
Example: Demonstrate time shift detection in load stage comparison

场景：工作日 vs 周末负荷对比
Scenario: Weekday vs Weekend load comparison

这个示例模拟周末起床时间推迟、早高峰后移2小时的情况
This example simulates weekend behavior with delayed wake-up time and morning peak shifted 2 hours later
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_workday_load():
    """生成工作日负荷曲线"""
    n_points = 96  # 24小时，每15分钟一个点
    load = np.zeros(n_points)
    
    # 夜间低负荷 (0-6h)
    load[0:24] = 0.5 + np.random.normal(0, 0.05, 24)
    
    # 早高峰 (6-9h) - 工作日起床、早餐
    load[24:36] = 2.5 + np.random.normal(0, 0.1, 12)
    
    # 上午到下午 (9-18h) - 大部分人外出
    load[36:72] = 1.0 + np.random.normal(0, 0.05, 36)
    
    # 晚高峰 (18-22h) - 回家、晚餐、娱乐
    load[72:88] = 3.0 + np.random.normal(0, 0.1, 16)
    
    # 夜间 (22-24h)
    load[88:96] = 0.8 + np.random.normal(0, 0.05, 8)
    
    return np.maximum(load, 0.1)  # 确保负荷为正

def generate_weekend_load():
    """生成周末负荷曲线 - 早高峰后移2小时"""
    n_points = 96
    load = np.zeros(n_points)
    
    # 夜间低负荷延长 (0-8h) - 周末睡懒觉，比工作日多睡2小时
    load[0:32] = 0.5 + np.random.normal(0, 0.05, 32)
    
    # 早高峰推迟到 (8-11h) - 起床时间推迟2小时
    load[32:44] = 2.8 + np.random.normal(0, 0.1, 12)  # 负荷稍高，因为周末在家准备早午餐
    
    # 白天 (11-18h) - 周末更多人在家
    load[44:72] = 1.5 + np.random.normal(0, 0.08, 28)  # 比工作日高
    
    # 晚高峰 (18-22h) - 时间基本不变
    load[72:88] = 3.2 + np.random.normal(0, 0.1, 16)  # 周末娱乐活动更多
    
    # 夜间 (22-24h) - 周末可能晚睡
    load[88:96] = 1.0 + np.random.normal(0, 0.05, 8)  # 比工作日稍高
    
    return np.maximum(load, 0.1)

def run_time_shift_demo():
    """运行时间偏移检测演示"""
    print("="*80)
    print("负荷阶段时间偏移检测演示")
    print("Demonstration: Time Shift Detection in Load Stage Comparison")
    print("="*80)
    print("\n场景说明 (Scenario Description):")
    print("  历史数据: 工作日负荷模式 (早高峰 6-9h)")
    print("  Historical: Weekday load pattern (morning peak 6-9h)")
    print("  当前数据: 周末负荷模式 (早高峰 8-11h，推迟2小时)")
    print("  Current: Weekend load pattern (morning peak 8-11h, delayed by 2 hours)")
    print("="*80 + "\n")
    
    # 生成负荷数据
    workday_load = generate_workday_load()
    weekend_load = generate_weekend_load()
    
    # 生成时间序列
    start_time = datetime(2024, 1, 1, 0, 0)
    times = pd.date_range(start=start_time, periods=96, freq='15min')
    
    # 创建特征数据框（简化版本，只包含必要字段）
    workday_df = pd.DataFrame({
        'load': workday_load,
        'temperature_current': 15 + 5 * np.sin(np.linspace(0, 2*np.pi, 96))
    }, index=times)
    
    weekend_df = pd.DataFrame({
        'load': weekend_load,
        'temperature_current': 18 + 5 * np.sin(np.linspace(0, 2*np.pi, 96))
    }, index=times)
    
    # 手动定义负荷分段（模拟HMM分段结果）
    # 格式: (start_idx, end_idx, state, mean_load)
    
    print("📊 负荷分段分析...")
    print("-" * 80)
    
    # 工作日分段
    workday_segments = [
        (0, 23, 0, 0.5),    # 0-6h, 夜间低负荷
        (24, 35, 2, 2.5),   # 6-9h, 早高峰
        (36, 71, 1, 1.0),   # 9-18h, 白天中等负荷
        (72, 87, 3, 3.0),   # 18-22h, 晚高峰
        (88, 95, 0, 0.8)    # 22-24h, 夜间
    ]
    
    print(f"\n工作日负荷阶段数: {len(workday_segments)}")
    for i, (start, end, state, mean) in enumerate(workday_segments, 1):
        start_h = start * 15 / 60
        end_h = (end + 1) * 15 / 60
        print(f"  阶段{i}: {start_h:.1f}h-{end_h:.1f}h, 平均负荷={mean:.2f} kW")
    
    # 周末分段（早高峰推迟2小时）
    weekend_segments = [
        (0, 31, 0, 0.5),    # 0-8h, 夜间低负荷延长
        (32, 43, 2, 2.8),   # 8-11h, 早高峰推迟
        (44, 71, 1, 1.5),   # 11-18h, 白天中等负荷
        (72, 87, 3, 3.2),   # 18-22h, 晚高峰
        (88, 95, 0, 1.0)    # 22-24h, 夜间
    ]
    
    print(f"\n周末负荷阶段数: {len(weekend_segments)}")
    for i, (start, end, state, mean) in enumerate(weekend_segments, 1):
        start_h = start * 15 / 60
        end_h = (end + 1) * 15 / 60
        print(f"  阶段{i}: {start_h:.1f}h-{end_h:.1f}h, 平均负荷={mean:.2f} kW")
    
    # 进行历史对比分析
    print("\n" + "="*80)
    print("🔍 历史负荷对比分析 (Historical Load Comparison)")
    print("="*80)
    
    from historical_comparison_demo import compare_with_historical_stages_standalone
    
    comparison = compare_with_historical_stages_standalone(
        weekend_segments,  # 当前（周末）
        workday_segments,  # 历史（工作日）
        weekend_df,
        workday_df,
        times.tolist(),
        times.tolist(),
        weekend_load,
        workday_load
    )
    
    # 显示阶段数量变化
    print("\n▶ 阶段数量变化分析:")
    print("-" * 80)
    scc = comparison['stage_count_comparison']
    print(f"当前阶段数: {scc['current_count']} (周末)")
    print(f"历史阶段数: {scc['historical_count']} (工作日)")
    print(f"变化: {scc['change']:+d} 个阶段 ({scc['change_percent']:+.1f}%)")
    print(f"趋势: {scc['trend']}\n")
    if scc.get('reasons'):
        print("原因分析:")
        for reason in scc['reasons']:
            print(f"  {reason}")
    
    # 显示逐阶段对齐结果和时间偏移
    print("\n▶ 逐阶段对齐分析 (含时间偏移检测):")
    print("-" * 80)
    for aligned in comparison['aligned_stages']:
        print(f"\n当前阶段{aligned['current_stage']} ↔ 历史阶段{aligned['historical_stage']}:")
        print(f"  时间范围: {aligned['current_time_range']} (周末) vs {aligned['historical_time_range']} (工作日)")
        
        # 重点显示时间偏移
        if 'time_shift' in aligned:
            time_shift = aligned['time_shift']
            if abs(time_shift) >= 0.5:  # 显示超过30分钟的偏移
                shift_dir = '右移(推迟)' if time_shift > 0 else '左移(提前)'
                shift_symbol = '→' if time_shift > 0 else '←'
                print(f"  ⏰ 时间偏移: {abs(time_shift):.1f} 小时 {shift_symbol} ({shift_dir})")
        
        print(f"  负荷水平: {aligned['current_load']:.2f} kW (周末) vs {aligned['historical_load']:.2f} kW (工作日)")
        print(f"  负荷差异: {aligned['load_difference']:+.2f} kW ({aligned['load_difference_percent']:+.1f}%)")
    
    # 显示差异显著的阶段（包括时间偏移）
    if comparison['significant_differences']:
        print("\n▶ 差异显著的负荷阶段:")
        print("-" * 80)
        for diff in comparison['significant_differences']:
            print(f"\n阶段{diff['current_stage']} (周末时间: {diff['time_range']}, 工作日时间: {diff.get('historical_time_range', 'N/A')}):")
            
            # 时间偏移信息
            if 'time_shift' in diff and abs(diff['time_shift']) >= 1.0:
                print(f"  ⏰ 时间偏移: {abs(diff['time_shift']):.1f} 小时 ({diff['shift_direction']})")
            
            # 负荷变化信息
            print(f"  📊 负荷变化: {diff['load_change']:+.2f} kW ({diff['load_change_percent']:+.1f}%)")
            print(f"  📈 变化类型: {diff['change_type']}")
            
            # 行为解释
            if diff['explanations']:
                print(f"  💡 行为解释:")
                for exp in diff['explanations']:
                    print(f"      • {exp}")
    
    # 显示总体行为解释
    if comparison['behavior_explanations']:
        print("\n▶ 总体行为模式分析:")
        print("-" * 80)
        for exp in comparison['behavior_explanations']:
            print(f"  {exp}")
    
    print("\n" + "="*80)
    print("演示完成！")
    print("Demonstration Complete!")
    print("="*80)
    
    # 保存报告（简化版，不依赖train_household_forecast）
    output_path = '/tmp/time_shift_demo_report.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("负荷阶段时间偏移检测演示报告\n")
        f.write("Time Shift Detection Demo Report\n")
        f.write("="*80 + "\n\n")
        
        # 写入场景说明
        f.write("场景说明:\n")
        f.write("  历史数据: 工作日负荷模式 (早高峰 6-9h)\n")
        f.write("  当前数据: 周末负荷模式 (早高峰 8-11h，推迟2小时)\n\n")
        
        # 写入对比结果
        if comparison.get('stage_count_comparison'):
            scc = comparison['stage_count_comparison']
            f.write("▶ 阶段数量变化分析\n")
            f.write("-"*80 + "\n")
            f.write(f"当前阶段数: {scc['current_count']}\n")
            f.write(f"历史阶段数: {scc['historical_count']}\n")
            f.write(f"变化: {scc['change']:+d} 个阶段 ({scc['change_percent']:+.1f}%)\n")
            f.write(f"趋势: {scc['trend']}\n\n")
        
        # 写入阶段对齐结果
        if comparison.get('aligned_stages'):
            f.write("\n▶ 逐阶段对齐分析 (含时间偏移)\n")
            f.write("-"*80 + "\n")
            for aligned in comparison['aligned_stages']:
                f.write(f"\n当前阶段{aligned['current_stage']} ↔ 历史阶段{aligned['historical_stage']}:\n")
                f.write(f"  时间范围: {aligned['current_time_range']} (周末) vs {aligned['historical_time_range']} (工作日)\n")
                
                if 'time_shift' in aligned and abs(aligned['time_shift']) >= 0.5:
                    shift_dir = '右移(推迟)' if aligned['time_shift'] > 0 else '左移(提前)'
                    f.write(f"  ⏰ 时间偏移: {abs(aligned['time_shift']):.1f} 小时 ({shift_dir})\n")
                
                f.write(f"  负荷水平: {aligned['current_load']:.2f} kW (周末) vs {aligned['historical_load']:.2f} kW (工作日)\n")
                f.write(f"  负荷差异: {aligned['load_difference']:+.2f} kW ({aligned['load_difference_percent']:+.1f}%)\n")
        
        # 写入差异显著的阶段
        if comparison.get('significant_differences'):
            f.write("\n\n▶ 差异显著的负荷阶段\n")
            f.write("-"*80 + "\n")
            for diff in comparison['significant_differences']:
                f.write(f"\n阶段{diff['current_stage']}:\n")
                
                if 'time_shift' in diff and abs(diff['time_shift']) >= 1.0:
                    f.write(f"  ⏰ 时间偏移: {abs(diff['time_shift']):.1f} 小时 ({diff['shift_direction']})\n")
                
                f.write(f"  📊 负荷变化: {diff['load_change']:+.2f} kW ({diff['load_change_percent']:+.1f}%)\n")
                f.write(f"  💡 行为解释:\n")
                for exp in diff['explanations']:
                    f.write(f"      • {exp}\n")
        
        # 写入总体分析
        if comparison.get('behavior_explanations'):
            f.write("\n\n▶ 总体行为模式分析\n")
            f.write("-"*80 + "\n")
            for exp in comparison['behavior_explanations']:
                f.write(f"  {exp}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("报告生成完成\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ 详细报告已保存到: {output_path}")
    
    return comparison

if __name__ == '__main__':
    try:
        comparison = run_time_shift_demo()
        print("\n✅ 演示成功完成！")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
