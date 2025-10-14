"""
Simple test script for historical load comparison functionality
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_comparison():
    """Test basic historical comparison functionality"""
    print("Testing historical load comparison...")
    
    # Create simple test data
    n_points = 96
    times = pd.date_range('2024-01-01', periods=n_points, freq='15min')
    
    # Current load: higher in morning (6-9h)
    hours = np.arange(n_points) / 4
    current_load = np.zeros(n_points)
    for i, h in enumerate(hours):
        if h < 6:
            current_load[i] = 0.5
        elif h < 9:
            current_load[i] = 2.5  # High morning load
        elif h < 18:
            current_load[i] = 1.0
        elif h < 22:
            current_load[i] = 3.0
        else:
            current_load[i] = 0.8
    
    # Historical load: lower in morning
    historical_load = np.copy(current_load)
    for i, h in enumerate(hours):
        if 6 <= h < 9:
            historical_load[i] = 1.5  # Lower morning load
    
    # Create dataframes
    current_df = pd.DataFrame({
        'load': current_load,
        'temperature_current': 20 + np.random.randn(n_points),
        'humidity_current': 60 + np.random.randn(n_points),
        'cloudCover_current': 0.5 + 0.1 * np.random.randn(n_points)
    }, index=times)
    
    historical_df = pd.DataFrame({
        'load': historical_load,
        'temperature_current': 15 + np.random.randn(n_points),
        'humidity_current': 65 + np.random.randn(n_points),
        'cloudCover_current': 0.6 + 0.1 * np.random.randn(n_points)
    }, index=times)
    
    # Simple segmentation
    def simple_segment(load, n_seg=4):
        quantiles = np.linspace(0, 1, n_seg + 1)
        thresholds = np.quantile(load, quantiles)
        states = np.digitize(load, thresholds[1:-1])
        
        segments = []
        current_state = states[0]
        start_idx = 0
        
        for i in range(1, len(states)):
            if states[i] != current_state:
                end_idx = i - 1
                segment_load = np.mean(load[start_idx:i])
                segments.append((start_idx, end_idx, current_state, segment_load))
                start_idx = i
                current_state = states[i]
        
        segment_load = np.mean(load[start_idx:])
        segments.append((start_idx, len(states) - 1, current_state, segment_load))
        return segments
    
    current_segments = simple_segment(current_load)
    historical_segments = simple_segment(historical_load)
    
    print(f"✓ Current segments: {len(current_segments)}")
    print(f"✓ Historical segments: {len(historical_segments)}")
    
    # Test the comparison function
    try:
        # Import from the demo file to avoid tensorflow dependency
        from historical_comparison_demo import compare_with_historical_stages_standalone
        
        comparison = compare_with_historical_stages_standalone(
            current_segments, historical_segments,
            current_df, historical_df,
            times.tolist(), times.tolist(),
            current_load, historical_load
        )
        
        # Verify results
        assert 'stage_count_comparison' in comparison
        assert 'aligned_stages' in comparison
        assert 'significant_differences' in comparison
        assert 'behavior_explanations' in comparison
        
        scc = comparison['stage_count_comparison']
        print(f"✓ Stage count comparison: {scc['current_count']} vs {scc['historical_count']}")
        print(f"✓ Aligned stages: {len(comparison['aligned_stages'])}")
        print(f"✓ Significant differences: {len(comparison['significant_differences'])}")
        
        # Should find at least one significant difference (morning period)
        assert len(comparison['significant_differences']) > 0, "Should find significant differences"
        
        # Check that explanations are generated
        assert len(comparison['behavior_explanations']) > 0, "Should have behavior explanations"
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stage_alignment():
    """Test stage alignment logic"""
    print("\nTesting stage alignment logic...")
    
    # Create segments that should align
    current_segments = [
        (0, 23, 0, 0.5),    # 0-6h, low
        (24, 35, 1, 2.5),   # 6-9h, high
        (36, 71, 0, 1.0),   # 9-18h, medium
        (72, 87, 2, 3.0),   # 18-22h, high
        (88, 95, 0, 0.8)    # 22-24h, low
    ]
    
    historical_segments = [
        (0, 23, 0, 0.5),
        (24, 35, 0, 1.5),   # Different load level in morning
        (36, 71, 0, 1.0),
        (72, 87, 2, 3.0),
        (88, 95, 0, 0.8)
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
    
    # Check alignment
    aligned = comparison['aligned_stages']
    assert len(aligned) == len(current_segments), "Should align all segments"
    
    # Check that morning segment (stage 2) is identified as different
    sig_diffs = comparison['significant_differences']
    morning_diff_found = False
    for diff in sig_diffs:
        if 6 <= float(diff['time_range'].split('h')[0]) < 9:
            morning_diff_found = True
            # Check that explanation mentions morning period
            explanations = ' '.join(diff['explanations'])
            assert '早高峰' in explanations or 'morning' in explanations.lower()
            print(f"✓ Found morning difference with explanation")
            break
    
    print("✅ Stage alignment test passed!")
    return True

if __name__ == '__main__':
    success = True
    
    try:
        success = test_basic_comparison() and success
        success = test_stage_alignment() and success
        
        if success:
            print("\n" + "="*60)
            print("All tests passed successfully! ✅")
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
