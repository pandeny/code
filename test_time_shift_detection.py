"""
Test time shift detection in historical load comparison
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_time_shift_detection():
    """Test detection of stage time shifts (left/right movement)"""
    print("\nTesting time shift detection...")
    
    # Create segments where morning peak is shifted right by 2 hours
    # Historical: low (0-6h), morning peak (6-9h), midday (9-18h), evening (18-22h), night (22-24h)
    # Current: low (0-8h), morning peak (8-11h), midday (11-18h), evening (18-22h), night (22-24h)
    # This simulates weekend behavior where people wake up later
    
    current_segments = [
        (0, 31, 0, 0.5),    # 0-8h, low (extended sleep - 2h longer)
        (32, 43, 1, 2.5),   # 8-11h, high (morning peak shifted by 2h)
        (44, 71, 0, 1.0),   # 11-18h, medium
        (72, 87, 2, 3.0),   # 18-22h, high
        (88, 95, 0, 0.8)    # 22-24h, low
    ]
    
    historical_segments = [
        (0, 23, 0, 0.5),    # 0-6h, low
        (24, 35, 1, 2.5),   # 6-9h, high (morning peak)
        (36, 71, 0, 1.0),   # 9-18h, medium
        (72, 87, 2, 3.0),   # 18-22h, high
        (88, 95, 0, 0.8)    # 22-24h, low
    ]
    
    # Create dummy data
    n_points = 96
    times = pd.date_range('2024-01-01', periods=n_points, freq='15min')
    dummy_df = pd.DataFrame({
        'load': np.ones(n_points),
        'temperature_current': np.ones(n_points) * 20
    }, index=times)
    
    from historical_comparison_demo import compare_with_historical_stages_standalone
    
    comparison = compare_with_historical_stages_standalone(
        current_segments, historical_segments,
        dummy_df, dummy_df,
        times.tolist(), times.tolist(),
        np.ones(n_points), np.ones(n_points)
    )
    
    # Check that time_shift is calculated for aligned stages
    aligned = comparison['aligned_stages']
    assert len(aligned) > 0, "Should have aligned stages"
    
    # Check that time_shift field exists
    print("\n  Aligned stages and their time shifts:")
    for stage in aligned:
        assert 'time_shift' in stage, "Each aligned stage should have time_shift field"
        print(f"    Current stage {stage['current_stage']} ↔ Historical {stage['historical_stage']}: "
              f"time_shift = {stage['time_shift']:+.2f}h")
    
    # Check that at least one stage has significant time shift
    has_significant_shift = any(abs(s['time_shift']) >= 1.0 for s in aligned)
    assert has_significant_shift, "Should detect at least one stage with significant time shift"
    print(f"  ✓ Detected stages with significant time shift (>= 1h)")
    
    # Check significant_differences includes time shift information
    sig_diffs = comparison['significant_differences']
    print(f"\n  Found {len(sig_diffs)} significant differences")
    
    shift_detected = False
    for diff in sig_diffs:
        if 'time_shift' in diff and abs(diff['time_shift']) >= 1.0:
            assert 'shift_direction' in diff, "Should have shift_direction field"
            print(f"    ✓ Stage {diff['current_stage']}: shift = {diff['time_shift']:+.2f}h, direction: {diff['shift_direction']}")
            shift_detected = True
            
            # Check that explanations mention the time shift
            has_shift_explanation = any('推迟' in exp or '提前' in exp or '小时' in exp 
                                       for exp in diff['explanations'])
            if has_shift_explanation:
                print(f"      ✓ Has time shift explanation in behavior analysis")
    
    if shift_detected:
        print(f"  ✓ Time shift successfully detected and explained")
    
    # Check behavior_explanations includes shift pattern analysis
    behavior_exp = comparison['behavior_explanations']
    has_shift_pattern = any('时间偏移模式' in exp or '右移' in exp or '左移' in exp 
                           for exp in behavior_exp)
    if has_shift_pattern:
        print(f"  ✓ Behavior explanations include time shift pattern analysis")
        for exp in behavior_exp:
            if '时间偏移' in exp or '右移' in exp or '左移' in exp:
                print(f"      {exp}")
    
    print("\n✅ Time shift detection test passed!")
    return True

def test_left_shift_detection():
    """Test detection of leftward (earlier) time shifts"""
    print("\nTesting leftward time shift detection...")
    
    # Create segments where current is shifted 1.5 hours earlier than historical
    # Historical: evening peak at 18-22h
    # Current: evening peak at 16.5-20.5h (shifted left)
    current_segments = [
        (0, 23, 0, 0.5),    # 0-6h, low
        (24, 35, 1, 2.0),   # 6-9h, medium
        (36, 65, 0, 1.0),   # 9-16.5h, low
        (66, 81, 2, 3.0),   # 16.5-20.5h, high (evening peak shifted left)
        (82, 95, 0, 0.8)    # 20.5-24h, low
    ]
    
    historical_segments = [
        (0, 23, 0, 0.5),    # 0-6h, low
        (24, 35, 1, 2.0),   # 6-9h, medium
        (36, 71, 0, 1.0),   # 9-18h, low
        (72, 87, 2, 3.0),   # 18-22h, high (evening peak)
        (88, 95, 0, 0.8)    # 22-24h, low
    ]
    
    # Create dummy data
    n_points = 96
    times = pd.date_range('2024-01-01', periods=n_points, freq='15min')
    dummy_df = pd.DataFrame({
        'load': np.ones(n_points),
        'temperature_current': np.ones(n_points) * 20
    }, index=times)
    
    from historical_comparison_demo import compare_with_historical_stages_standalone
    
    comparison = compare_with_historical_stages_standalone(
        current_segments, historical_segments,
        dummy_df, dummy_df,
        times.tolist(), times.tolist(),
        np.ones(n_points), np.ones(n_points)
    )
    
    # Find evening peak stage
    evening_stage = next((s for s in comparison['aligned_stages'] if s['current_stage'] == 4), None)
    assert evening_stage is not None, "Should find evening stage"
    
    # The evening peak should have shifted left (negative time_shift)
    # Current evening center: (16.5 + 20.5) / 2 = 18.5h
    # Historical evening center: (18 + 22) / 2 = 20h
    # Expected shift: 18.5 - 20 = -1.5h
    assert evening_stage['time_shift'] < -1.0, f"Evening peak should shift left, got {evening_stage['time_shift']:.2f}h"
    print(f"  ✓ Evening peak shifted left by {abs(evening_stage['time_shift']):.2f}h")
    
    # Check for left shift in significant differences
    sig_diffs = [d for d in comparison['significant_differences'] 
                 if d.get('shift_direction') == '左移(提前)']
    if sig_diffs:
        print(f"  ✓ Detected {len(sig_diffs)} stages with leftward shift")
    
    print("✅ Leftward shift detection test passed!")
    return True

if __name__ == '__main__':
    success = True
    
    try:
        success = test_time_shift_detection() and success
        success = test_left_shift_detection() and success
        
        if success:
            print("\n" + "="*60)
            print("All time shift detection tests passed! ✅")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("Some tests failed ❌")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
