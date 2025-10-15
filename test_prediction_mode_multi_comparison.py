"""
测试预测模式中的多历史时期对比分析功能
"""

import sys
import os

# Mock tensorflow and keras modules before any imports
import types

# Create mock tensorflow
tf_mock = types.ModuleType('tensorflow')
tf_mock.random = types.ModuleType('random')
tf_mock.random.set_seed = lambda x: None
tf_mock.keras = types.ModuleType('keras')
tf_mock.keras.layers = types.ModuleType('layers')
tf_mock.keras.models = types.ModuleType('models')
tf_mock.keras.models.load_model = lambda x: None

sys.modules['tensorflow'] = tf_mock
sys.modules['tensorflow.keras'] = tf_mock.keras
sys.modules['tensorflow.keras.layers'] = tf_mock.keras.layers
sys.modules['tensorflow.keras.models'] = tf_mock.keras.models

# Now safe to import other modules
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

# Now import functions
from train_household_forecast import (
    prepare_historical_data_for_comparison,
    compare_predicted_with_multiple_historical_stages,
    generate_multi_historical_comparison_report,
    hmm_load_segmentation,
    simple_load_segmentation
)

def generate_test_data(date_str, scenario='weekday', n_points=96):
    """生成测试数据"""
    times = pd.date_range(date_str, periods=n_points, freq='15min')
    hours = np.arange(n_points) / 4
    load = np.zeros(n_points)
    
    # 根据场景生成不同的负荷模式
    for i, h in enumerate(hours):
        if scenario == 'weekday':
            if h < 6:
                base = 0.5
            elif h < 8:
                progress = (h - 6) / 2
                base = 0.5 + progress * 2.0
            elif h < 18:
                base = 0.8
            elif h < 22:
                progress = (h - 18) / 4
                base = 0.8 + progress * 2.5
            else:
                base = 2.5
        else:  # weekend
            if h < 8:
                base = 0.5
            elif h < 10:
                progress = (h - 8) / 2
                base = 0.5 + progress * 2.5
            elif h < 18:
                base = 1.8
            elif h < 23:
                progress = (h - 18) / 5
                base = 1.8 + progress * 2.0
            else:
                base = 3.0
        
        load[i] = base + np.random.normal(0, 0.05)
    
    # 确保负荷为正值
    load = np.maximum(load, 0.3)
    
    # 生成环境特征
    base_temp = {'weekday': 15, 'weekend': 18}.get(scenario, 15)
    temperature = base_temp + 8 * np.sin((hours - 6) * np.pi / 12)
    humidity = 60 + 12 * np.cos((hours - 3) * np.pi / 12)
    cloudCover = np.clip(0.3 + 0.3 * np.sin(hours * np.pi / 24), 0, 1)
    
    df = pd.DataFrame({
        'load': load,
        'temperature': temperature,
        'humidity': humidity,
        'cloudCover': cloudCover,
        'hour': hours
    }, index=times)
    
    return df

def test_prepare_historical_data():
    """测试历史数据准备功能"""
    print("="*80)
    print("测试：历史数据准备功能")
    print("="*80)
    
    # 生成测试数据 - 连续8天的数据
    all_data_list = []
    for i in range(8):
        date_str = (datetime(2024, 1, 14) - timedelta(days=i)).strftime('%Y-%m-%d')
        scenario = 'weekend' if i in [0, 1, 7] else 'weekday'
        df = generate_test_data(date_str, scenario=scenario)
        all_data_list.append(df)
    
    # 合并所有数据
    ts = pd.concat(all_data_list, axis=0).sort_index()
    feat_df = ts.copy()
    feat_df['target_next'] = feat_df['load'].shift(-1)
    
    target_date = datetime(2024, 1, 14)
    
    print(f"\n生成的测试数据时间范围: {ts.index.min()} 到 {ts.index.max()}")
    print(f"目标预测日期: {target_date.date()}")
    
    # 测试准备历史数据
    print("\n调用 prepare_historical_data_for_comparison...")
    historical_data_dict = prepare_historical_data_for_comparison(
        ts, feat_df, target_date, comparison_days=[1, 3, 7]
    )
    
    print(f"\n✅ 成功准备 {len(historical_data_dict)} 个历史时期的数据")
    
    for days_ago, hist_data in historical_data_dict.items():
        print(f"\n{days_ago}天前:")
        print(f"  阶段数: {len(hist_data['segments'])}")
        print(f"  数据点数: {len(hist_data['load'])}")
        print(f"  时间范围: {hist_data['times'][0]} 到 {hist_data['times'][-1]}")
    
    return True

def test_multi_historical_comparison():
    """测试多历史时期对比分析功能"""
    print("\n" + "="*80)
    print("测试：多历史时期对比分析功能")
    print("="*80)
    
    # 生成测试数据
    all_data_list = []
    for i in range(8):
        date_str = (datetime(2024, 1, 14) - timedelta(days=i)).strftime('%Y-%m-%d')
        scenario = 'weekend' if i in [0, 1, 7] else 'weekday'
        df = generate_test_data(date_str, scenario=scenario)
        all_data_list.append(df)
    
    ts = pd.concat(all_data_list, axis=0).sort_index()
    feat_df = ts.copy()
    feat_df['target_next'] = feat_df['load'].shift(-1)
    
    target_date = datetime(2024, 1, 14)
    
    # 准备预测日数据
    pred_date_obj = target_date.date()
    pred_day_data = ts[ts.index.date == pred_date_obj]
    pred_load = pred_day_data['load'].values
    pred_times = pred_day_data.index.tolist()
    
    # 对预测负荷进行阶段划分
    print("\n对预测日负荷进行阶段划分...")
    try:
        _, _, pred_segments = hmm_load_segmentation(pred_load, n_states='auto', min_states=3, max_states=5)
    except:
        _, _, pred_segments = simple_load_segmentation(pred_load, n_segments=4)
    
    print(f"预测日识别出 {len(pred_segments)} 个阶段")
    
    # 准备历史数据
    print("\n准备历史数据...")
    historical_data_dict = prepare_historical_data_for_comparison(
        ts, feat_df, target_date, comparison_days=[1, 3, 7]
    )
    
    if not historical_data_dict:
        print("❌ 未能准备历史数据")
        return False
    
    # 执行多历史时期对比分析
    print("\n执行多历史时期对比分析...")
    multi_comparison = compare_predicted_with_multiple_historical_stages(
        pred_segments,
        historical_data_dict,
        pred_day_data,
        pred_times,
        pred_load,
        comparison_days=[1, 3, 7]
    )
    
    print("\n✅ 多历史时期对比分析完成")
    
    # 检查结果结构
    assert 'comparison_days' in multi_comparison, "缺少 comparison_days 字段"
    assert 'comparisons' in multi_comparison, "缺少 comparisons 字段"
    assert 'summary' in multi_comparison, "缺少 summary 字段"
    
    print(f"\n对比结果包含 {len(multi_comparison['comparisons'])} 个历史时期")
    
    # 打印阶段数量变化趋势
    if multi_comparison['summary'].get('stage_count_trends'):
        print("\n阶段数量变化趋势:")
        for sc in multi_comparison['summary']['stage_count_trends']:
            print(f"  {sc['days_ago']}天前: 预测{sc['predicted_count']}个阶段 vs 历史{sc['historical_count']}个阶段, 变化{sc['change']:+d}")
    
    # 打印行为模式总结
    if multi_comparison['summary'].get('behavior_patterns'):
        print("\n行为模式总结:")
        for pattern in multi_comparison['summary']['behavior_patterns']:
            print(f"  • {pattern}")
    
    # 保存报告
    output_path = '/tmp/test_multi_historical_comparison_report.txt'
    print(f"\n生成报告到 {output_path}...")
    generate_multi_historical_comparison_report(multi_comparison, output_path)
    
    # 检查报告文件是否生成
    assert os.path.exists(output_path), "报告文件未生成"
    
    # 读取并打印部分报告内容
    with open(output_path, 'r', encoding='utf-8') as f:
        report_lines = f.readlines()[:30]  # 只读取前30行
    
    print("\n报告前30行内容:")
    print("".join(report_lines))
    
    print("\n✅ 测试通过！")
    return True

if __name__ == '__main__':
    try:
        print("开始测试...\n")
        
        # 测试1: 历史数据准备
        test_prepare_historical_data()
        
        # 测试2: 多历史时期对比分析
        test_multi_historical_comparison()
        
        print("\n" + "="*80)
        print("所有测试通过！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
