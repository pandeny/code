#!/usr/bin/env python3
"""
测试峰值/波动检测的负荷阶段划分功能
"""
import numpy as np
import sys

def test_peak_detection():
    """测试峰值检测功能"""
    print("=" * 80)
    print("测试峰值/波动检测的负荷阶段划分")
    print("=" * 80)
    
    # 生成模拟数据：96个点（24小时，15分钟间隔）
    # 包含两个明显的峰值：早高峰和晚高峰
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
    
    # 添加一些噪声
    np.random.seed(42)
    load_values += np.random.normal(0, 0.05, n_points)
    load_values = np.maximum(load_values, 0.1)  # 确保非负
    
    print(f"\n📊 生成测试数据:")
    print(f"   数据点数: {n_points}")
    print(f"   负荷范围: {load_values.min():.3f} - {load_values.max():.3f} kW")
    print(f"   平均负荷: {load_values.mean():.3f} kW")
    
    # 简单检测峰值区域（不依赖scipy）
    from collections import Counter
    
    # 手动实现median_filter
    def simple_median_filter(data, size=3):
        filtered = np.zeros_like(data)
        half_size = size // 2
        for i in range(len(data)):
            start = max(0, i - half_size)
            end = min(len(data), i + half_size + 1)
            filtered[i] = np.median(data[start:end])
        return filtered
    
    smoothed_load = simple_median_filter(load_values, size=3)
    
    # 检测峰值
    window_size = 8
    peak_zones = []
    fluctuation_zones = []
    
    for i in range(window_size, n_points - window_size):
        window = smoothed_load[i-window_size:i+window_size]
        center_val = smoothed_load[i]
        
        # 峰值检测
        if center_val == np.max(window) and center_val > np.percentile(smoothed_load, 75):
            peak_zones.append(i)
        
        # 波动检测
        window_std = np.std(window)
        if window_std > np.std(smoothed_load) * 0.8:
            fluctuation_zones.append(i)
    
    print(f"\n🔍 检测结果:")
    print(f"   检测到峰值点: {len(peak_zones)} 个")
    print(f"   检测到波动点: {len(fluctuation_zones)} 个")
    
    if peak_zones:
        peak_hours = [p * 0.25 for p in peak_zones]
        print(f"   峰值时段: {min(peak_hours):.1f}h - {max(peak_hours):.1f}h")
    
    # 测试features列表组装
    print(f"\n🧮 测试时间特征组装:")
    features = []
    time_features = []
    for i in range(n_points):
        hour = (i * 0.25) % 24
        time_features.append([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * (i % 96) / 96),
            np.cos(2 * np.pi * (i % 96) / 96)
        ])
    features.append(np.array(time_features))
    time_features_array = features[0]
    
    print(f"   features列表长度: {len(features)}")
    print(f"   time_features形状: {time_features_array.shape}")
    print(f"   时间特征维度: {time_features_array.shape[1]}")
    
    # 验证时间连续性
    # 检查23:45和00:00的时间特征相似度
    idx_2345 = 95  # 23:45
    idx_0000 = 0   # 00:00
    # 使用归一化的余弦相似度
    vec1 = time_features_array[idx_2345]
    vec2 = time_features_array[idx_0000]
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"   23:45与00:00的相似度: {similarity:.4f} (应接近1.0)")
    
    print(f"\n✅ 测试完成!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        success = test_peak_detection()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
