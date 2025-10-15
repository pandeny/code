#!/usr/bin/env python3
"""
测试实际的简单负荷分段函数
"""
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def test_simple_load_segmentation():
    """测试simple_load_segmentation函数"""
    print("=" * 80)
    print("测试 simple_load_segmentation 函数")
    print("=" * 80)
    
    # 生成测试数据：双峰模式
    n_points = 96
    hours = np.arange(n_points) * 0.25
    
    # 基础负荷
    base_load = 0.5
    
    # 早高峰 (6-9h)
    morning_peak = 1.5 * np.exp(-((hours - 7.5)**2) / (2 * 1.0**2))
    
    # 晚高峰 (18-22h)
    evening_peak = 2.5 * np.exp(-((hours - 20)**2) / (2 * 1.5**2))
    
    # 白天基础负荷
    day_load = 0.3 * (1 + np.sin(2 * np.pi * (hours - 6) / 24))
    
    # 组合负荷
    load_values = base_load + morning_peak + evening_peak + day_load
    
    # 添加噪声
    np.random.seed(42)
    load_values += np.random.normal(0, 0.05, n_points)
    load_values = np.maximum(load_values, 0.1)
    
    print(f"\n📊 测试数据:")
    print(f"   数据点数: {n_points}")
    print(f"   负荷范围: {load_values.min():.3f} - {load_values.max():.3f} kW")
    print(f"   平均负荷: {load_values.mean():.3f} kW")
    
    # 导入并测试函数
    try:
        from train_household_forecast import simple_load_segmentation
        print(f"\n✅ 成功导入 simple_load_segmentation")
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        return False
    
    # 执行分段
    try:
        states, state_means, segments = simple_load_segmentation(load_values, n_segments=4)
        print(f"\n🎯 分段结果:")
        print(f"   状态数量: {len(np.unique(states))}")
        print(f"   段落数量: {len(segments)}")
        print(f"   状态均值: {state_means}")
        
        print(f"\n📋 段落详情:")
        for i, (start_idx, end_idx, state, mean_load) in enumerate(segments):
            start_hour = start_idx * 0.25
            end_hour = (end_idx + 1) * 0.25
            duration = (end_idx - start_idx + 1) * 0.25
            print(f"   阶段{i+1}: {start_hour:.1f}h - {end_hour:.1f}h "
                  f"(时长{duration:.1f}h, 平均负荷{mean_load:.3f}kW, 状态{state})")
        
        # 验证基本要求
        assert len(segments) >= 3, "段落数量应该至少为3个"
        assert len(segments) <= 8, "段落数量不应超过8个"
        
        # 验证段落覆盖所有时间点
        total_points = sum(end - start + 1 for start, end, _, _ in segments)
        assert total_points == n_points, f"段落应该覆盖所有时间点: {total_points} vs {n_points}"
        
        # 验证段落是连续的
        for i in range(len(segments) - 1):
            assert segments[i][1] + 1 == segments[i+1][0], f"段落{i}和{i+1}不连续"
        
        print(f"\n✅ 所有验证通过!")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 80)
    return True

if __name__ == "__main__":
    try:
        success = test_simple_load_segmentation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
