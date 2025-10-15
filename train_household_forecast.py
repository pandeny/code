#往添加环境特征方向走
"""
家庭负荷预测脚本（CNN、LSTM、GRU）
说明：
- 从文件夹读取某户负荷CSV，默认使用文件夹中第一个CSV文件。
- 将原始时间序列聚合为日尺度特征：均值、标准差、最大值、最小值、斜率。
- 使用前3个月（约90天）样本训练，后3个月样本验证；若数据不足则退化为70/30分割。
- 使用滑动窗口构建序列，预测下一日的日均负荷（回归）。
- 输出每个模型在验证集上的MAPE和RMSE。
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import re
import random
import warnings
from sklearn.mixture import GaussianMixture

# 尝试导入hmmlearn，如果不可用则使用替代方案
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️ hmmlearn 未安装，将使用简单分段方法作为替代")

# 抑制特定的RuntimeWarning
warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation*')
# 或者抑制所有scipy统计相关警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats')
# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 添加中文字体配置
import matplotlib
from matplotlib import font_manager

# 配置
DATA_FOLDER = r"D:\python project\xuheng-homeloadprediction\OutputData\combined\all"
MODEL_OUTPUT_DIR = r"D:\python project\负荷预测项目\output\model"
ANALYSIS_OUTPUT_DIR = r"D:\python project\负荷预测项目\output\analysis"
SEQ_LEN = 14  # 用过去14天预测下一天
EPOCHS = 50
BATCH_SIZE = 64
RANDOM_SEED = 42

# 若需指定绘图的日期（格式 'YYYY-MM-DD'），可设置 PLOT_DATE；默认为 None（使用验证集中的第一条样本日期）
PLOT_DATE = None

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 检查是否有中文字体
HAS_CJK_FONT = True
try:
    # 尝试使用中文字体
    test_font = font_manager.FontProperties(family='SimHei')
    if not test_font:
        HAS_CJK_FONT = False
except:
    HAS_CJK_FONT = False

# 确保输出目录存在
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)


def find_csv_file(folder):
    files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"在 {folder} 未找到 CSV 文件")
    files.sort()
    return files[0]

# 新增：返回文件夹中所有 CSV 列表（按路径排序）
def find_csv_files(folder):
    files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"在 {folder} 未找到 CSV 文件")
    files.sort()
    return files

def find_apartment_files(folder):
    """查找所有公寓数据文件（Apt*_2015.csv格式）并按户号排序"""
    pattern = os.path.join(folder, "Apt*_2015.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"在 {folder} 未找到 Apt*_2015.csv 格式的文件")
    
    # 按户号排序
    def extract_apt_number(filename):
        import re
        basename = os.path.basename(filename)
        match = re.search(r'Apt(\d+)_2015\.csv', basename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_apt_number)
    return files

def extract_household_name(csv_path):
    """从CSV文件路径提取家庭名称"""
    basename = os.path.basename(csv_path)
    # 去掉文件扩展名
    name = os.path.splitext(basename)[0]
    return name

def interactive_select_households():
    """交互式选择要训练的户数"""
    print("\n🏠 家庭负荷预测训练 - 单户选择")
    print("="*60)
    
    # 查找所有公寓文件
    try:
        all_files = find_apartment_files(DATA_FOLDER)
        print(f"📊 发现 {len(all_files)} 户数据文件")
        
        # 显示前10个和后10个文件作为示例
        print("\n📋 可用户数据文件示例：")
        if len(all_files) <= 20:
            for i, file in enumerate(all_files, 1):
                basename = os.path.basename(file)
                print(f"  {i:3d}. {basename}")
        else:
            for i in range(10):
                basename = os.path.basename(all_files[i])
                print(f"  {i+1:3d}. {basename}")
            print("  ...")
            for i in range(len(all_files)-10, len(all_files)):
                basename = os.path.basename(all_files[i])
                print(f"  {i+1:3d}. {basename}")
        
        print(f"\n🎯 请选择要训练的用户：")
        print("输入方式：")
        print("  - 户号：直接输入户号（如 1, 2, 1114）")
        print("  - 序号：输入文件序号（如 1-114，对应上面列表中的序号）")
        
        while True:
            try:
                selection = input("\n请输入户号或文件序号: ").strip()
                
                if not selection.isdigit():
                    print("❌ 请输入有效的数字")
                    continue
                
                num = int(selection)
                selected_files = []
                
                if num <= len(all_files):
                    # 作为序号处理
                    selected_files = [all_files[num-1]]
                    print(f"✅ 按文件序号选择: {os.path.basename(selected_files[0])}")
                else:
                    # 作为户号处理，查找对应文件
                    target_file = None
                    for file in all_files:
                        if f"Apt{num}_2015.csv" in os.path.basename(file):
                            target_file = file
                            break
                    if target_file:
                        selected_files = [target_file]
                        print(f"✅ 按户号选择: {os.path.basename(selected_files[0])}")
                    else:
                        print(f"❌ 未找到户号 {num} 的数据文件")
                        continue
                
                # 确认选择并输入自定义模型名称
                confirm = input(f"\n确认使用 {os.path.basename(selected_files[0])} 进行训练？(y/n): ").strip().lower()
                if confirm in ['y', 'yes', '是', '']:
                    # 让用户输入自定义模型名称
                    while True:
                        model_name = input(f"\n请输入模型名称（用于保存模型文件）: ").strip()
                        if not model_name:
                            print("❌ 模型名称不能为空")
                            continue
                        # 检查名称是否合法（避免特殊字符）
                        import re
                        if not re.match(r'^[a-zA-Z0-9_\-\u4e00-\u9fa5]+$', model_name):
                            print("❌ 模型名称只能包含字母、数字、下划线、连字符和中文字符")
                            continue
                        break
                    return selected_files, 'single', model_name
                else:
                    print("请重新选择...")
                    continue
                        
            except ValueError:
                print("❌ 输入格式错误，请输入数字")
                continue
            except KeyboardInterrupt:
                print("\n\n❌ 用户取消操作")
                return [], 'cancel', None
            except Exception as e:
                print(f"❌ 选择过程出错: {e}")
                continue
                
    except Exception as e:
        print(f"❌ 查找数据文件失败: {e}")
        return [], 'error', None


def load_time_series(csv_path):
    # 尝试读取并解析时间列、负荷列和环境特征列
    df = pd.read_csv(csv_path, encoding='utf-8')

    print(f"📊 数据文件列名: {df.columns.tolist()}")

    # 识别时间列
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'datetime' in c.lower()]
    if time_cols:
        tcol = time_cols[0]
    else:
        tcol = df.columns[0]

    # 识别负荷列（通常是Value列）
    load_col = None
    if 'Value' in df.columns:
        load_col = 'Value'
    elif 'value' in df.columns:
        load_col = 'value'
    else:
        # 备用方案：寻找数值列
        num_cols = [c for c in df.columns if c != tcol and np.issubdtype(df[c].dtype, np.number)]
        if num_cols:
            load_col = num_cols[0]
        else:
            raise ValueError('无法识别负荷数值列')

    # 定义需要的环境特征列
    env_features = ['temperature', 'humidity', 'visibility', 'pressure', 'windSpeed', 'cloudCover', 'dewPoint']
    
    # 检查哪些环境特征列存在
    available_env_features = []
    for feat in env_features:
        if feat in df.columns:
            available_env_features.append(feat)
    
    print(f"📊 发现环境特征: {available_env_features}")

    # 选择需要的列
    selected_cols = [tcol, load_col] + available_env_features
    
    # 确保所有需要的列都存在
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 缺少列: {missing_cols}")
        # 移除缺失的列
        selected_cols = [col for col in selected_cols if col in df.columns]

    # 转换数据类型
    for col in selected_cols[1:]:  # 跳过时间列
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 解析时间并设置索引
    try:
        df[tcol] = pd.to_datetime(df[tcol])
    except Exception:
        # 若无法解析时间，则按行号生成时间（每15分钟）
        df[tcol] = pd.date_range(start='2020-01-01', periods=len(df), freq='15T')

    # 只保留选择的列
    df = df[selected_cols].copy()
    
    # 温度转换：华氏度转摄氏度
    if 'temperature' in df.columns:
        df['temperature'] = (df['temperature'] - 32) * 5/9
        print("🌡️ 已将温度从华氏度转换为摄氏度")

    # 处理缺失值和异常值
    df = df.dropna(subset=[load_col])  # 移除负荷数据缺失的行
    
    # 对环境特征的缺失值进行插值
    for feat in available_env_features:
        if df[feat].isna().any():
            df[feat] = df[feat].interpolate(method='time').fillna(method='ffill').fillna(method='bfill')

    df = df.sort_values(tcol).set_index(tcol)

    # 重命名负荷列
    rename_dict = {load_col: 'load'}
    df = df.rename(columns=rename_dict)

    print(f"✅ 成功加载数据，包含 {len(df)} 行，{len(df.columns)} 个特征")
    print(f"📊 特征列: {df.columns.tolist()}")

    return df

1
def smooth_series(series, sigma=2.0):
    arr = series.to_numpy().astype(float)
    return gaussian_filter1d(arr, sigma=sigma)


def build_time_features(df, sigma=2.0, seq_len=96):
    """
    对原始按时间序列的负荷数据做平滑并提取特征，同时包含环境特征。
    假设原始时间间隔为15分钟（每日96个点），若不同请调整lags。
    返回特征DataFrame（与原始索引对齐，不再过早丢弃前部样本；特征不足处置为 NaN，后续统一填充）。
    """
    arr = df['load'].astype(float)
    smooth = pd.Series(smooth_series(arr, sigma=sigma), index=df.index)

    # 检查哪些环境特征可用
    env_features = ['temperature', 'humidity', 'visibility', 'pressure', 'windSpeed', 'cloudCover', 'dewPoint']
    available_env_features = [feat for feat in env_features if feat in df.columns]
    
    print(f"📊 构建特征时使用的环境特征: {available_env_features}")

    n = len(smooth)
    # 定义滞后步长（15min分辨率）
    lag_1d = 96
    lag_7d = lag_1d * 7

    records = []
    idxs = []

    for i in range(n):
        cur_idx = df.index[i]
        cur_val = smooth.iloc[i]

        feat = {}
        # 原始与平滑值
        feat['load_smooth'] = float(cur_val)
        feat['load_raw'] = float(arr.iloc[i])

        # 滞后特征（如果历史不足则为 NaN）
        feat['lag_1d'] = float(smooth.iloc[i - lag_1d]) if i - lag_1d >= 0 else np.nan
        feat['lag_7d'] = float(smooth.iloc[i - lag_7d]) if i - lag_7d >= 0 else np.nan

        # 过去7天同期中位数（取最近7个相同时间点）
        same_times = []
        for d in range(1, 8):
            idx = i - d * lag_1d
            if idx >= 0:
                same_times.append(smooth.iloc[idx])
        feat['median_past7_same_time'] = float(np.median(same_times)) if same_times else np.nan

        # 过去1小时（4个点）与1天（96个点）的统计特征
        h1_start = max(0, i - 4)
        h1_window = smooth.iloc[h1_start:i] if i > 0 else smooth.iloc[0:0]
        d1_start = max(0, i - lag_1d)
        d1_window = smooth.iloc[d1_start:i] if i > 0 else smooth.iloc[0:0]

        feat['h1_mean'] = float(h1_window.mean()) if len(h1_window) > 0 else np.nan
        feat['h1_var'] = float(h1_window.var()) if len(h1_window) > 0 else np.nan
        feat['h1_kurtosis'] = float(kurtosis(h1_window)) if len(h1_window) > 0 else np.nan

        feat['d1_mean'] = float(d1_window.mean()) if len(d1_window) > 0 else np.nan
        feat['d1_var'] = float(d1_window.var()) if len(d1_window) > 0 else np.nan
        feat['d1_kurtosis'] = float(kurtosis(d1_window)) if len(d1_window) > 0 else np.nan

        # 最大负载率（窗口最大/窗口平均）
        feat['h1_max_load_rate'] = float(h1_window.max() / (h1_window.mean() + 1e-8)) if len(h1_window) > 0 else np.nan
        feat['d1_max_load_rate'] = float(d1_window.max() / (d1_window.mean() + 1e-8)) if len(d1_window) > 0 else np.nan

        # 频域特征 - 对过去一天窗口做FFT，取前几个幅值
        fft_window = d1_window if len(d1_window) >= 8 else smooth.iloc[max(0, i - 32):i]
        if len(fft_window) > 3:
            fft_vals = np.abs(np.fft.rfft(fft_window.values - np.mean(fft_window.values)))
            # 取前3个频率幅值（除直流）
            fft_vals = fft_vals[1:4] if len(fft_vals) > 3 else np.pad(fft_vals[1:], (0, max(0, 3 - len(fft_vals[1:]))), 'constant')
            # 可能因长度问题导致单元素数组，需要安全索引
            fft_padded = np.pad(fft_vals, (0, 3 - len(fft_vals)), 'constant')
            feat['fft_1'] = float(fft_padded[0])
            feat['fft_2'] = float(fft_padded[1])
            feat['fft_3'] = float(fft_padded[2])
        else:
            feat['fft_1'] = feat['fft_2'] = feat['fft_3'] = 0.0

        # 时间相关特征（小时、分钟）
        feat['hour'] = float(cur_idx.hour)
        feat['minute'] = float(cur_idx.minute)

        # 添加环境特征
        for env_feat in available_env_features:
            # 当前时刻的环境特征
            feat[f'{env_feat}_current'] = float(df[env_feat].iloc[i])
            
            # 过去1小时平均值
            h1_env_start = max(0, i - 4)
            h1_env_window = df[env_feat].iloc[h1_env_start:i] if i > 0 else df[env_feat].iloc[0:0]
            feat[f'{env_feat}_h1_mean'] = float(h1_env_window.mean()) if len(h1_env_window) > 0 else feat[f'{env_feat}_current']
            
            # 过去1天平均值
            d1_env_start = max(0, i - lag_1d)
            d1_env_window = df[env_feat].iloc[d1_env_start:i] if i > 0 else df[env_feat].iloc[0:0]
            feat[f'{env_feat}_d1_mean'] = float(d1_env_window.mean()) if len(d1_env_window) > 0 else feat[f'{env_feat}_current']
            
            # 滞后特征（1天前）
            feat[f'{env_feat}_lag_1d'] = float(df[env_feat].iloc[i - lag_1d]) if i - lag_1d >= 0 else feat[f'{env_feat}_current']

        # 目标值（下一时刻的平滑负荷），用于监督学习
        if i + 1 < n:
            feat['target_next'] = float(smooth.iloc[i + 1])
        else:
            feat['target_next'] = np.nan

        records.append(feat)
        idxs.append(cur_idx)

    feat_df = pd.DataFrame(records, index=pd.DatetimeIndex(idxs))
    # 丢弃包含NaN目标的尾部（最后一条通常没有下一时刻目标）
    feat_df = feat_df.dropna(subset=['target_next'])
    
    print(f"📊 最终特征维度: {feat_df.shape[1]-1} 个特征（不包括目标变量）")
    print(f"📊 特征名称: {[col for col in feat_df.columns if col != 'target_next']}")
    
    return feat_df


def build_sequences_from_features(feat_df, seq_days=1, step_per_day=96):
    """
    构建LSTM输入序列：以时间序列顺序用过去 seq_days*step_per_day 个时刻的特征预测下一个时刻目标。
    默认不做左端 padding，只有历史长度 >= seq_len 时才构建样本，避免使用重复填充导致模型学习到常数模式。
    若确实需要保留早期样本，可将下面的逻辑改回 padding 策略或实现更合理的填充（如基于窗口均值或镜像填充）。
    """
    seq_len = seq_days * step_per_day
    X, y, dates = [], [], []
    values = feat_df.values
    cols = feat_df.columns.tolist()
    target_idx = cols.index('target_next')

    n = len(values)
    for i in range(n):
        # 仅在有足够历史（完整窗口）时构建样本，避免左侧重复填充
        start = i - seq_len
        if start < 0:
            # 跳过早期样本
            continue
        seq = values[start:i, :target_idx]

        X.append(seq)
        y.append(values[i, target_idx])
        dates.append(feat_df.index[i])

    X = np.array(X)
    y = np.array(y)
    return X, y, dates


def time_order_split(X, y, dates, train_frac=0.7, test_frac=0.15):
    n = len(X)
    n_train = int(n * train_frac)
    n_test = int(n * test_frac)
    n_val = n - n_train - n_test
    X_train = X[:n_train]; y_train = y[:n_train]
    X_test = X[n_train:n_train + n_test]; y_test = y[n_train:n_train + n_test]
    X_val = X[n_train + n_test:]; y_val = y[n_train + n_test:]
    dates_train = dates[:n_train]; dates_test = dates[n_train:n_train + n_test]; dates_val = dates[n_train + n_test:]
    return (X_train, y_train, dates_train), (X_test, y_test, dates_test), (X_val, y_val, dates_val)


def build_lstm_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=False)(inp)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model


def crps_gaussian(y_true, mu, sigma):
    # CRPS for Gaussian forecast with mean mu and std sigma
    # vectorized implementation
    z = (y_true - mu) / (sigma + 1e-12)
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / math.sqrt(math.pi))
    return np.mean(crps)


def coverage_probability(y_true, mu, sigma, alpha=0.95):
    z = norm.ppf((1 + alpha) / 2.0)
    lower = mu - z * sigma
    upper = mu + z * sigma
    return np.mean((y_true >= lower) & (y_true <= upper))


def mbe(y_true, y_pred):
    return np.mean(y_pred - y_true)


def hmm_load_segmentation(load_values, n_states='auto', min_states=3, max_states=5, min_segment_length=8):
    """
    使用隐马尔可夫模型对负荷曲线进行智能分段（优化版本，减少过度分段）
    
    参数:
    - load_values: 负荷值序列（一维数组）
    - n_states: 状态数量，'auto'为自动选择，或指定整数
    - min_states: 自动选择时的最小状态数
    - max_states: 自动选择时的最大状态数
    - min_segment_length: 最小段长度（时间点数）
    
    返回:
    - states: 每个时间点对应的状态序列
    - state_means: 每个状态的平均负荷水平
    - segments: 连续段信息 [(start_idx, end_idx, state, mean_load), ...]
    """
    # 如果hmmlearn不可用，直接使用简单分段方法
    if not HMM_AVAILABLE:
        return simple_load_segmentation(load_values, n_segments=4)
    
    try:
        # 数据预处理
        load_values = np.array(load_values).reshape(-1, 1)
        
        # 自动选择最优状态数（减少最大状态数）
        if n_states == 'auto':
            best_score = -np.inf
            best_n_states = min_states
            
            for n in range(min_states, max_states + 1):
                try:
                    # 使用GaussianHMM进行训练
                    model = hmm.GaussianHMM(n_components=n, covariance_type="full", random_state=42)
                    model.fit(load_values)
                    score = model.score(load_values)
                    
                    if score > best_score:
                        best_score = score
                        best_n_states = n
                except:
                    continue
            
            n_states = best_n_states
        
        # 训练最终的HMM模型
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", random_state=42)
        
        # 设置更高的自转移概率，减少状态切换
        transition_prob = 0.98  # 提高保持当前状态的概率
        transfer_prob = (1 - transition_prob) / (n_states - 1)  # 转移到其他状态的概率
        
        transmat = np.full((n_states, n_states), transfer_prob)
        np.fill_diagonal(transmat, transition_prob)
        model.transmat_ = transmat
        
        # 训练模型
        model.fit(load_values)
        
        # 预测状态序列
        raw_states = model.predict(load_values)
        
        # 应用中值滤波减少噪声
        from scipy import ndimage
        smoothed_states = ndimage.median_filter(raw_states.astype(float), size=5).astype(int)
        
        # 计算每个状态的平均负荷水平
        state_means = []
        for state in range(n_states):
            state_mask = (smoothed_states == state)
            if np.any(state_mask):
                state_mean = np.mean(load_values[state_mask])
                state_means.append(state_mean)
            else:
                state_means.append(0)
        
        state_means = np.array(state_means).flatten()
        
        # 根据平均负荷水平对状态进行排序（从低到高）
        sorted_indices = np.argsort(state_means)
        state_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        
        # 重新映射状态
        mapped_states = np.array([state_mapping[s] for s in smoothed_states])
        sorted_state_means = state_means[sorted_indices]
        
        # 识别初始连续段
        initial_segments = []
        current_state = mapped_states[0]
        start_idx = 0
        
        for i in range(1, len(mapped_states)):
            if mapped_states[i] != current_state:
                # 当前段结束
                end_idx = i - 1
                segment_load = np.mean(load_values[start_idx:i])
                initial_segments.append((start_idx, end_idx, current_state, segment_load))
                
                # 开始新段
                start_idx = i
                current_state = mapped_states[i]
        
        # 添加最后一段
        segment_load = np.mean(load_values[start_idx:])
        initial_segments.append((start_idx, len(mapped_states) - 1, current_state, segment_load))
        
        # 后处理：合并短小段落
        merged_segments = merge_short_segments(initial_segments, load_values, min_segment_length)
        
        # 重新构建状态序列
        final_states = np.zeros_like(mapped_states)
        for start_idx, end_idx, state, _ in merged_segments:
            final_states[start_idx:end_idx+1] = state
        
        # 重新计算状态平均值
        final_state_means = []
        unique_states = sorted(set([seg[2] for seg in merged_segments]))
        for state in unique_states:
            state_mask = (final_states == state)
            if np.any(state_mask):
                state_mean = np.mean(load_values[state_mask])
                final_state_means.append(state_mean)
            else:
                final_state_means.append(0)
        
        final_state_means = np.array(final_state_means).flatten()
        
        return final_states, final_state_means, merged_segments
        
    except Exception as e:
        print(f"❌ HMM分段失败: {e}")
        # 退回到简单的分段方法
        return simple_load_segmentation(load_values.flatten(), n_segments=4)

def merge_short_segments(segments, load_values, min_segment_length=8):
    """
    合并过短的段落，减少过度分割
    
    参数:
    - segments: 初始段落列表 [(start, end, state, mean_load), ...]
    - load_values: 负荷值数组
    - min_segment_length: 最小段落长度
    
    返回:
    - merged_segments: 合并后的段落列表
    """
    if len(segments) <= 1:
        return segments
    
    merged = []
    i = 0
    
    while i < len(segments):
        start_idx, end_idx, state, mean_load = segments[i]
        segment_length = end_idx - start_idx + 1
        
        # 如果当前段太短，尝试与相邻段合并
        if segment_length < min_segment_length and len(merged) > 0:
            # 与前一个段合并
            prev_start, prev_end, prev_state, prev_mean = merged[-1]
            
            # 计算合并后的平均负荷
            combined_load = np.mean(load_values[prev_start:end_idx+1])
            
            # 决定使用哪个状态（选择负荷水平更接近合并后平均值的状态）
            if abs(prev_mean - combined_load) <= abs(mean_load - combined_load):
                final_state = prev_state
            else:
                final_state = state
            
            # 更新最后一个段
            merged[-1] = (prev_start, end_idx, final_state, combined_load)
            
        elif segment_length < min_segment_length and i < len(segments) - 1:
            # 与下一个段合并
            next_start, next_end, next_state, next_mean = segments[i + 1]
            
            # 计算合并后的平均负荷
            combined_load = np.mean(load_values[start_idx:next_end+1])
            
            # 决定使用哪个状态
            if abs(mean_load - combined_load) <= abs(next_mean - combined_load):
                final_state = state
            else:
                final_state = next_state
            
            # 添加合并后的段
            merged.append((start_idx, next_end, final_state, combined_load))
            i += 1  # 跳过下一个段，因为已经合并了
            
        else:
            # 段长度足够，直接添加
            merged.append((start_idx, end_idx, state, mean_load))
        
        i += 1
    
    # 如果还有很短的段，进行二次合并
    if len(merged) > 1:
        final_merged = []
        for seg in merged:
            start_idx, end_idx, state, mean_load = seg
            segment_length = end_idx - start_idx + 1
            
            if segment_length < min_segment_length // 2 and len(final_merged) > 0:
                # 与前一个段合并
                prev_start, prev_end, prev_state, prev_mean = final_merged[-1]
                combined_load = np.mean(load_values[prev_start:end_idx+1])
                final_merged[-1] = (prev_start, end_idx, prev_state, combined_load)
            else:
                final_merged.append(seg)
        
        return final_merged
    
    return merged

def explain_load_changes(segments, feat_df, pred_times, load_values):
    """
    负荷变化可解释性模型 - 分析负荷阶段变化的原因
    
    参数:
    - segments: 负荷分段信息 [(start_idx, end_idx, state, mean_load), ...]
    - feat_df: 特征数据框
    - pred_times: 预测时间点列表
    - load_values: 负荷值数组
    
    返回:
    - explanations: 包含各阶段变化解释的字典
    """
    try:
        explanations = {
            'segment_analysis': [],
            'trend_analysis': {},
            'feature_importance': {},
            'environmental_impact': {}
        }
        
        # 检查可用的环境特征
        env_features = ['temperature', 'humidity', 'visibility', 'pressure', 'windSpeed', 'cloudCover', 'dewPoint']
        available_env_features = []
        for feat in env_features:
            if f'{feat}_current' in feat_df.columns:
                available_env_features.append(feat)
        
        # 1. 逐段分析负荷特征和变化原因
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
            
            # 提取该段的特征数据（如果时间对齐）
            if pred_times and len(pred_times) > end_idx:
                try:
                    # 获取该段时间范围内的特征数据
                    segment_times = pred_times[start_idx:end_idx+1]
                    
                    # 查找特征数据中对应的时间点
                    matching_features = []
                    for t in segment_times:
                        # 找到最接近的特征数据时间点
                        idx = feat_df.index.get_indexer([t], method='nearest')[0]
                        if 0 <= idx < len(feat_df):
                            matching_features.append(feat_df.iloc[idx])
                    
                    if matching_features:
                        # 计算该段的平均特征值
                        segment_features = pd.DataFrame(matching_features)
                        
                        # 分析环境因素的影响
                        for env_feat in available_env_features:
                            current_col = f'{env_feat}_current'
                            if current_col in segment_features.columns:
                                avg_value = segment_features[current_col].mean()
                                
                                # 根据特征值判断影响
                                if env_feat == 'temperature':
                                    if avg_value > 25:
                                        segment_info['key_factors'].append(f'高温({avg_value:.1f}°C)可能增加空调负荷')
                                    elif avg_value < 10:
                                        segment_info['key_factors'].append(f'低温({avg_value:.1f}°C)可能增加供暖负荷')
                                    else:
                                        segment_info['key_factors'].append(f'温度适中({avg_value:.1f}°C)')
                                
                                elif env_feat == 'humidity':
                                    if avg_value > 70:
                                        segment_info['key_factors'].append(f'高湿度({avg_value:.1f}%)可能增加除湿需求')
                                    elif avg_value < 30:
                                        segment_info['key_factors'].append(f'低湿度({avg_value:.1f}%)')
                                
                                elif env_feat == 'cloudCover':
                                    if avg_value > 0.7:
                                        segment_info['key_factors'].append(f'多云({avg_value:.2f})减少自然采光')
                                    elif avg_value < 0.3:
                                        segment_info['key_factors'].append(f'晴朗({avg_value:.2f})增加自然采光')
                        
                        # 分析时间特征的影响
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
                        
                        # 分析负荷变化率
                        if 'load_smooth' in segment_features.columns:
                            load_std = segment_features['load_smooth'].std()
                            if load_std > 0.2:
                                segment_info['key_factors'].append(f'负荷波动较大(标准差={load_std:.3f})')
                            else:
                                segment_info['key_factors'].append(f'负荷相对稳定(标准差={load_std:.3f})')
                
                except Exception as e:
                    print(f"⚠️ 分析阶段 {i+1} 特征时出错: {e}")
            
            # 如果没有找到具体因素，添加通用说明
            if not segment_info['key_factors']:
                segment_info['key_factors'].append('负荷水平主要由用户行为模式决定')
            
            explanations['segment_analysis'].append(segment_info)
        
        # 2. 分析阶段间的趋势变化
        if len(segments) > 1:
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
                
                # 尝试解释变化原因（基于时间和环境）
                curr_start_hour = curr_seg[0] * 15 / 60
                next_start_hour = next_seg[0] * 15 / 60
                
                # 时间相关的变化解释
                if curr_start_hour < 6 and next_start_hour >= 6:
                    trend_info['explanation'].append('进入早晨时段，家庭活动增加')
                elif curr_start_hour < 18 and next_start_hour >= 18:
                    trend_info['explanation'].append('进入傍晚时段，家庭成员返回')
                elif curr_start_hour < 22 and next_start_hour >= 22:
                    trend_info['explanation'].append('进入深夜时段，活动减少')
                elif curr_start_hour >= 9 and next_start_hour < 18:
                    trend_info['explanation'].append('日间时段，多数家庭成员外出工作')
                
                trend_changes.append(trend_info)
            
            explanations['trend_analysis'] = {
                'total_segments': len(segments),
                'transitions': trend_changes,
                'max_load': float(max([seg[3] for seg in segments])),
                'min_load': float(min([seg[3] for seg in segments])),
                'load_range': float(max([seg[3] for seg in segments]) - min([seg[3] for seg in segments]))
            }
        
        # 3. 特征重要性分析（基于特征变化与负荷变化的相关性）
        if available_env_features:
            feature_correlations = {}
            
            for env_feat in available_env_features:
                current_col = f'{env_feat}_current'
                if current_col in feat_df.columns:
                    try:
                        # 简单的相关性分析
                        if 'load_smooth' in feat_df.columns:
                            valid_indices = ~(feat_df[current_col].isna() | feat_df['load_smooth'].isna())
                            if valid_indices.sum() > 10:
                                correlation = feat_df.loc[valid_indices, current_col].corr(
                                    feat_df.loc[valid_indices, 'load_smooth']
                                )
                                feature_correlations[env_feat] = float(correlation)
                    except Exception as e:
                        print(f"⚠️ 计算 {env_feat} 相关性时出错: {e}")
            
            # 排序特征重要性
            sorted_features = sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            explanations['feature_importance'] = {
                'correlations': dict(sorted_features),
                'top_features': [f[0] for f in sorted_features[:3]],
                'interpretation': []
            }
            
            for feat, corr in sorted_features[:3]:
                if abs(corr) > 0.3:
                    direction = '正相关' if corr > 0 else '负相关'
                    explanations['feature_importance']['interpretation'].append(
                        f'{feat}与负荷呈{direction}(相关系数={corr:.3f})'
                    )
        
        # 4. 环境因素综合影响评估
        if available_env_features and pred_times:
            env_impact = {}
            
            for env_feat in available_env_features:
                current_col = f'{env_feat}_current'
                if current_col in feat_df.columns:
                    try:
                        # 计算全天该特征的统计信息
                        values = []
                        for t in pred_times:
                            idx = feat_df.index.get_indexer([t], method='nearest')[0]
                            if 0 <= idx < len(feat_df):
                                val = feat_df.iloc[idx][current_col]
                                if not np.isnan(val):
                                    values.append(val)
                        
                        if values:
                            env_impact[env_feat] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values)),
                                'min': float(np.min(values)),
                                'max': float(np.max(values)),
                                'range': float(np.max(values) - np.min(values))
                            }
                    except Exception as e:
                        print(f"⚠️ 分析 {env_feat} 影响时出错: {e}")
            
            explanations['environmental_impact'] = env_impact
        
        return explanations
        
    except Exception as e:
        print(f"❌ 负荷变化解释分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'segment_analysis': [],
            'trend_analysis': {},
            'feature_importance': {},
            'environmental_impact': {},
            'error': str(e)
        }

def compare_with_historical_stages(current_segments, historical_segments, 
                                   current_feat_df, historical_feat_df,
                                   current_times, historical_times,
                                   current_load, historical_load):
    """
    与历史负荷阶段进行对比分析
    
    参数:
    - current_segments: 当前负荷分段信息 [(start_idx, end_idx, state, mean_load), ...]
    - historical_segments: 历史负荷分段信息 [(start_idx, end_idx, state, mean_load), ...]
    - current_feat_df: 当前特征数据框
    - historical_feat_df: 历史特征数据框
    - current_times: 当前时间点列表
    - historical_times: 历史时间点列表
    - current_load: 当前负荷值数组
    - historical_load: 历史负荷值数组
    
    返回:
    - comparison: 包含对比分析结果的字典
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
        
        # 2. 逐阶段对齐分析（基于时间和负荷水平）
        # 使用简单的时间重叠匹配策略
        aligned_pairs = []
        
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
                aligned_pairs.append((curr_idx, best_match))
                
                # 计算负荷差异
                load_diff = curr_mean - hist_mean
                load_diff_pct = (load_diff / hist_mean * 100) if hist_mean != 0 else 0
                
                # 提取环境特征差异
                env_diff = {}
                try:
                    # 获取当前阶段的环境特征
                    if current_times and len(current_times) > curr_end:
                        curr_segment_times = current_times[curr_start:curr_end+1]
                        curr_features = []
                        for t in curr_segment_times:
                            idx = current_feat_df.index.get_indexer([t], method='nearest')[0]
                            if 0 <= idx < len(current_feat_df):
                                curr_features.append(current_feat_df.iloc[idx])
                        
                        if curr_features:
                            curr_feat_mean = pd.DataFrame(curr_features).mean()
                            
                            # 获取历史阶段的环境特征
                            if historical_times and len(historical_times) > hist_end:
                                hist_segment_times = historical_times[hist_start:hist_end+1]
                                hist_features = []
                                for t in hist_segment_times:
                                    idx = historical_feat_df.index.get_indexer([t], method='nearest')[0]
                                    if 0 <= idx < len(historical_feat_df):
                                        hist_features.append(historical_feat_df.iloc[idx])
                                
                                if hist_features:
                                    hist_feat_mean = pd.DataFrame(hist_features).mean()
                                    
                                    # 计算主要环境特征的差异
                                    for feat in ['temperature_current', 'humidity_current', 'cloudCover_current']:
                                        if feat in curr_feat_mean.index and feat in hist_feat_mean.index:
                                            env_diff[feat] = {
                                                'current': float(curr_feat_mean[feat]),
                                                'historical': float(hist_feat_mean[feat]),
                                                'diff': float(curr_feat_mean[feat] - hist_feat_mean[feat])
                                            }
                except Exception as e:
                    print(f"⚠️ 提取环境特征差异时出错: {e}")
                
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
                    'environment_diff': env_diff
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
                    if aligned['load_difference'] > 0:  # 负荷增加
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
                        else:  # 夜间
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
                        else:  # 夜间
                            diff_info['explanations'].append(
                                '夜间时段负荷减少，可能是：就寝时间提前、关闭更多电器、或减少设备待机功耗'
                            )
                
                # 结合环境特征的解释
                if 'environment_diff' in aligned and aligned['environment_diff']:
                    env_diff = aligned['environment_diff']
                    
                    if 'temperature_current' in env_diff:
                        temp_diff = env_diff['temperature_current']['diff']
                        if abs(temp_diff) > 5:
                            if temp_diff > 0:
                                diff_info['explanations'].append(
                                    f'环境温度升高{abs(temp_diff):.1f}°C，可能增加空调制冷需求'
                                )
                            else:
                                diff_info['explanations'].append(
                                    f'环境温度降低{abs(temp_diff):.1f}°C，可能增加供暖需求'
                                )
                    
                    if 'humidity_current' in env_diff:
                        humid_diff = env_diff['humidity_current']['diff']
                        if abs(humid_diff) > 15:
                            if humid_diff > 0:
                                diff_info['explanations'].append(
                                    f'湿度增加{abs(humid_diff):.1f}%，可能影响除湿设备使用'
                                )
                            else:
                                diff_info['explanations'].append(
                                    f'湿度降低{abs(humid_diff):.1f}%，可能减少除湿需求'
                                )
                    
                    if 'cloudCover_current' in env_diff:
                        cloud_diff = env_diff['cloudCover_current']['diff']
                        if abs(cloud_diff) > 0.3:
                            if cloud_diff > 0:
                                diff_info['explanations'].append(
                                    '云量增加，自然采光减少，可能增加照明负荷'
                                )
                            else:
                                diff_info['explanations'].append(
                                    '云量减少，自然采光增加，可能减少照明负荷'
                                )
                
                comparison['significant_differences'].append(diff_info)
        
        # 5. 生成总体行为解释
        if comparison['significant_differences']:
            comparison['behavior_explanations'].append(
                f"共识别出{len(comparison['significant_differences'])}个差异显著的负荷阶段"
            )
            
            # 统计增加和减少的阶段数
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

def compare_predicted_with_multiple_historical_stages(predicted_segments, historical_data_dict,
                                                      predicted_feat_df, predicted_times, predicted_load,
                                                      comparison_days=[1, 3, 7]):
    """
    将预测日负荷与多个历史负荷（7/3/1天前）进行对比分析
    
    参数:
    - predicted_segments: 预测日负荷分段信息 [(start_idx, end_idx, state, mean_load), ...]
    - historical_data_dict: 历史数据字典，格式为 {days_ago: {'segments': [...], 'feat_df': df, 'times': [...], 'load': [...]}}
    - predicted_feat_df: 预测日特征数据框
    - predicted_times: 预测日时间点列表
    - predicted_load: 预测日负荷值数组
    - comparison_days: 要对比的历史天数列表，默认为[1, 3, 7]天前
    
    返回:
    - multi_comparison: 包含与多个历史时期对比分析结果的字典
    """
    try:
        multi_comparison = {
            'comparison_days': comparison_days,
            'comparisons': {},
            'summary': {
                'stage_count_trends': [],
                'load_trends': [],
                'time_shift_trends': [],
                'behavior_patterns': []
            }
        }
        
        # 对每个历史时期进行对比
        for days_ago in comparison_days:
            if days_ago not in historical_data_dict:
                print(f"⚠️ {days_ago}天前的历史数据不存在，跳过")
                continue
            
            hist_data = historical_data_dict[days_ago]
            
            # 调用单次对比函数
            comparison = compare_with_historical_stages(
                predicted_segments, hist_data['segments'],
                predicted_feat_df, hist_data['feat_df'],
                predicted_times, hist_data['times'],
                predicted_load, hist_data['load']
            )
            
            multi_comparison['comparisons'][days_ago] = comparison
        
        # 生成跨时期的趋势总结
        if len(multi_comparison['comparisons']) > 0:
            # 1. 阶段数量变化趋势
            stage_counts = []
            for days_ago in sorted(multi_comparison['comparisons'].keys()):
                comp = multi_comparison['comparisons'][days_ago]
                scc = comp.get('stage_count_comparison', {})
                stage_counts.append({
                    'days_ago': days_ago,
                    'predicted_count': scc.get('current_count', 0),
                    'historical_count': scc.get('historical_count', 0),
                    'change': scc.get('change', 0)
                })
            
            multi_comparison['summary']['stage_count_trends'] = stage_counts
            
            # 分析阶段数量变化趋势
            if len(stage_counts) > 1:
                changes = [sc['change'] for sc in stage_counts]
                if all(c > 0 for c in changes):
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与过去{comparison_days}天相比，负荷阶段数持续增加，说明用电行为逐渐多样化和复杂化'
                    )
                elif all(c < 0 for c in changes):
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与过去{comparison_days}天相比，负荷阶段数持续减少，说明用电行为逐渐规律化和简化'
                    )
                else:
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与过去{comparison_days}天相比，负荷阶段数变化不一致，说明用电模式存在波动'
                    )
            
            # 2. 负荷变化趋势分析
            for days_ago in sorted(multi_comparison['comparisons'].keys()):
                comp = multi_comparison['comparisons'][days_ago]
                sig_diffs = comp.get('significant_differences', [])
                
                if sig_diffs:
                    increase_count = sum(1 for d in sig_diffs if d['load_change'] > 0)
                    decrease_count = len(sig_diffs) - increase_count
                    
                    multi_comparison['summary']['load_trends'].append({
                        'days_ago': days_ago,
                        'increase_count': increase_count,
                        'decrease_count': decrease_count,
                        'total_significant': len(sig_diffs)
                    })
            
            # 3. 时间偏移趋势分析
            for days_ago in sorted(multi_comparison['comparisons'].keys()):
                comp = multi_comparison['comparisons'][days_ago]
                sig_diffs = comp.get('significant_differences', [])
                
                if sig_diffs:
                    right_shift_count = sum(1 for d in sig_diffs if d.get('time_shift', 0) >= 1.0)
                    left_shift_count = sum(1 for d in sig_diffs if d.get('time_shift', 0) <= -1.0)
                    shift_count = right_shift_count + left_shift_count
                    
                    if shift_count > 0:
                        multi_comparison['summary']['time_shift_trends'].append({
                            'days_ago': days_ago,
                            'shift_count': shift_count,
                            'right_shift_count': right_shift_count,
                            'left_shift_count': left_shift_count,
                            'dominant_direction': '右移(推迟)' if right_shift_count > left_shift_count else ('左移(提前)' if left_shift_count > right_shift_count else '混合')
                        })
            
            # 4. 生成综合行为模式解释
            if multi_comparison['summary']['time_shift_trends']:
                # 分析时间偏移的一致性
                shift_trends = multi_comparison['summary']['time_shift_trends']
                dominant_dirs = [st['dominant_direction'] for st in shift_trends]
                
                if all('右移' in d for d in dominant_dirs):
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与过去{comparison_days}天相比，负荷阶段持续右移(推迟)，说明作息时间逐渐推迟，可能是周末/假日效应、或生活习惯改变'
                    )
                elif all('左移' in d for d in dominant_dirs):
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与过去{comparison_days}天相比，负荷阶段持续左移(提前)，说明作息时间逐渐提前，可能是工作日效应、或早起习惯养成'
                    )
                else:
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与过去{comparison_days}天相比，负荷阶段时间偏移方向不一致，说明作息时间有波动，用电模式不稳定'
                    )
            
            # 5. 负荷水平变化趋势
            if multi_comparison['summary']['load_trends']:
                load_trends = multi_comparison['summary']['load_trends']
                
                # 计算整体负荷增减趋势
                total_increases = sum(lt['increase_count'] for lt in load_trends)
                total_decreases = sum(lt['decrease_count'] for lt in load_trends)
                
                if total_increases > total_decreases * 1.5:
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与历史时期相比，负荷整体呈上升趋势，可能原因：在家时间增加、新增用电设备、季节性需求上升'
                    )
                elif total_decreases > total_increases * 1.5:
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与历史时期相比，负荷整体呈下降趋势，可能原因：外出时间增加、节能措施实施、季节性需求下降'
                    )
                else:
                    multi_comparison['summary']['behavior_patterns'].append(
                        f'与历史时期相比，负荷增减较为平衡，用电水平相对稳定'
                    )
        
        return multi_comparison
        
    except Exception as e:
        print(f"❌ 多历史时期对比分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'comparison_days': comparison_days,
            'comparisons': {},
            'summary': {},
            'error': str(e)
        }

def generate_explanation_report(explanations, output_path):
    """
    生成负荷变化可解释性报告（文本格式）
    
    参数:
    - explanations: 解释分析结果字典
    - output_path: 报告保存路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("负荷变化可解释性分析报告\n")
            f.write("="*80 + "\n\n")
            
            # 1. 阶段分析
            f.write("【阶段详细分析】\n")
            f.write("-"*80 + "\n")
            for seg in explanations.get('segment_analysis', []):
                f.write(f"\n阶段 {seg['segment_id']}:\n")
                f.write(f"  时间范围: {seg['start_time']:.2f}h - {seg['end_time']:.2f}h (持续 {seg['duration_hours']:.2f}小时)\n")
                f.write(f"  负荷水平: {seg['load_level']} (平均值: {seg['mean_load']:.4f})\n")
                f.write(f"  状态编号: {seg['state']}\n")
                f.write(f"  关键影响因素:\n")
                for factor in seg['key_factors']:
                    f.write(f"    • {factor}\n")
            
            # 2. 趋势分析
            if explanations.get('trend_analysis'):
                f.write("\n\n【阶段间趋势变化分析】\n")
                f.write("-"*80 + "\n")
                trend = explanations['trend_analysis']
                f.write(f"总阶段数: {trend.get('total_segments', 0)}\n")
                f.write(f"负荷范围: {trend.get('min_load', 0):.4f} - {trend.get('max_load', 0):.4f}\n")
                f.write(f"负荷波动幅度: {trend.get('load_range', 0):.4f}\n\n")
                
                for trans in trend.get('transitions', []):
                    f.write(f"\n阶段 {trans['from_segment']} → 阶段 {trans['to_segment']}:\n")
                    f.write(f"  变化趋势: {trans['trend']}\n")
                    f.write(f"  负荷变化: {trans['load_change']:+.4f} ({trans['load_change_percent']:+.1f}%)\n")
                    f.write(f"  变化原因:\n")
                    for exp in trans['explanation']:
                        f.write(f"    • {exp}\n")
            
            # 3. 特征重要性
            if explanations.get('feature_importance'):
                f.write("\n\n【特征重要性分析】\n")
                f.write("-"*80 + "\n")
                feat_imp = explanations['feature_importance']
                
                if feat_imp.get('top_features'):
                    f.write("最重要的环境特征:\n")
                    for feat in feat_imp['top_features']:
                        corr = feat_imp['correlations'].get(feat, 0)
                        f.write(f"  • {feat} (相关系数: {corr:+.3f})\n")
                
                if feat_imp.get('interpretation'):
                    f.write("\n特征影响解释:\n")
                    for interp in feat_imp['interpretation']:
                        f.write(f"  • {interp}\n")
            
            # 4. 环境因素影响
            if explanations.get('environmental_impact'):
                f.write("\n\n【环境因素综合影响】\n")
                f.write("-"*80 + "\n")
                for feat, stats in explanations['environmental_impact'].items():
                    f.write(f"\n{feat}:\n")
                    f.write(f"  平均值: {stats['mean']:.2f}\n")
                    f.write(f"  标准差: {stats['std']:.2f}\n")
                    f.write(f"  范围: {stats['min']:.2f} - {stats['max']:.2f}\n")
                    f.write(f"  波动幅度: {stats['range']:.2f}\n")
            
            # 5. 历史对比分析（如果有）
            if explanations.get('historical_comparison'):
                comp = explanations['historical_comparison']
                f.write("\n\n【与历史负荷对比分析】\n")
                f.write("="*80 + "\n")
                
                # 5.1 阶段数量对比
                if comp.get('stage_count_comparison'):
                    scc = comp['stage_count_comparison']
                    f.write("\n▶ 阶段数量变化分析\n")
                    f.write("-"*80 + "\n")
                    f.write(f"当前阶段数: {scc['current_count']}\n")
                    f.write(f"历史阶段数: {scc['historical_count']}\n")
                    f.write(f"变化: {scc['change']:+d} 个阶段 ({scc['change_percent']:+.1f}%)\n")
                    f.write(f"趋势: {scc['trend']}\n\n")
                    f.write("原因分析:\n")
                    for reason in scc.get('reasons', []):
                        f.write(f"  {reason}\n")
                
                # 5.2 逐阶段对齐分析
                if comp.get('aligned_stages'):
                    f.write("\n\n▶ 逐阶段对齐分析\n")
                    f.write("-"*80 + "\n")
                    for aligned in comp['aligned_stages']:
                        f.write(f"\n当前阶段{aligned['current_stage']} ↔ 历史阶段{aligned['historical_stage']}:\n")
                        f.write(f"  时间范围: {aligned['current_time_range']} (当前) vs {aligned['historical_time_range']} (历史)\n")
                        
                        # 添加时间偏移信息
                        if 'time_shift' in aligned:
                            time_shift = aligned['time_shift']
                            if abs(time_shift) >= 1.0:
                                shift_dir = '右移(推迟)' if time_shift > 0 else '左移(提前)'
                                f.write(f"  时间偏移: {abs(time_shift):.1f} 小时 ({shift_dir})\n")
                        
                        f.write(f"  负荷水平: {aligned['current_load']:.4f} kW (当前) vs {aligned['historical_load']:.4f} kW (历史)\n")
                        f.write(f"  负荷差异: {aligned['load_difference']:+.4f} kW ({aligned['load_difference_percent']:+.1f}%)\n")
                        if aligned.get('environment_diff'):
                            f.write(f"  环境特征差异:\n")
                            for feat, diff in aligned['environment_diff'].items():
                                feat_name = feat.replace('_current', '')
                                f.write(f"    • {feat_name}: {diff['current']:.2f} (当前) vs {diff['historical']:.2f} (历史), 差异: {diff['diff']:+.2f}\n")
                
                # 5.3 显著差异阶段
                if comp.get('significant_differences'):
                    f.write("\n\n▶ 差异显著的负荷阶段\n")
                    f.write("-"*80 + "\n")
                    for diff in comp['significant_differences']:
                        f.write(f"\n阶段{diff['current_stage']} (当前时间: {diff['time_range']}, 历史时间: {diff['historical_time_range']}):\n")
                        f.write(f"  负荷变化: {diff['load_change']:+.4f} kW ({diff['load_change_percent']:+.1f}%)\n")
                        f.write(f"  变化类型: {diff['change_type']}\n")
                        
                        # 添加时间偏移信息
                        if 'time_shift' in diff and abs(diff['time_shift']) >= 1.0:
                            f.write(f"  时间偏移: {diff['time_shift']:+.1f} 小时 ({diff['shift_direction']})\n")
                        
                        f.write(f"  行为解释:\n")
                        for exp in diff['explanations']:
                            f.write(f"    • {exp}\n")
                
                # 5.4 总体行为解释
                if comp.get('behavior_explanations'):
                    f.write("\n\n▶ 总体行为模式分析\n")
                    f.write("-"*80 + "\n")
                    for exp in comp['behavior_explanations']:
                        f.write(f"{exp}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("报告生成完成\n")
            f.write("="*80 + "\n")
        
        print(f"✅ 可解释性报告已保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 生成报告失败: {e}")

def visualize_explanations(explanations, output_path):
    """
    可视化负荷变化解释结果
    
    参数:
    - explanations: 解释分析结果字典
    - output_path: 图片保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        # 创建多子图布局
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. 阶段负荷水平柱状图
        ax1 = fig.add_subplot(gs[0, :])
        seg_analysis = explanations.get('segment_analysis', [])
        if seg_analysis:
            seg_ids = [seg['segment_id'] for seg in seg_analysis]
            seg_loads = [seg['mean_load'] for seg in seg_analysis]
            seg_levels = [seg['load_level'] for seg in seg_analysis]
            
            colors = []
            for level in seg_levels:
                if level == '低负荷':
                    colors.append('green')
                elif level == '中低负荷':
                    colors.append('lightgreen')
                elif level == '中高负荷':
                    colors.append('orange')
                else:
                    colors.append('red')
            
            bars = ax1.bar(seg_ids, seg_loads, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('阶段编号' if HAS_CJK_FONT else 'Stage ID', fontsize=12)
            ax1.set_ylabel('平均负荷' if HAS_CJK_FONT else 'Average Load', fontsize=12)
            ax1.set_title('各阶段负荷水平对比' if HAS_CJK_FONT else 'Load Level Comparison', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, load, level in zip(bars, seg_loads, seg_levels):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{load:.3f}\n{level}',
                        ha='center', va='bottom', fontsize=9)
        
        # 2. 趋势变化折线图
        ax2 = fig.add_subplot(gs[1, 0])
        trend_analysis = explanations.get('trend_analysis', {})
        if trend_analysis and trend_analysis.get('transitions'):
            transitions = trend_analysis['transitions']
            from_segs = [t['from_segment'] for t in transitions]
            change_pcts = [t['load_change_percent'] for t in transitions]
            
            ax2.plot(from_segs, change_pcts, marker='o', linewidth=2, markersize=8, color='blue')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.fill_between(from_segs, 0, change_pcts, alpha=0.3, 
                            color=['green' if c < 0 else 'red' for c in change_pcts])
            ax2.set_xlabel('起始阶段' if HAS_CJK_FONT else 'From Stage', fontsize=12)
            ax2.set_ylabel('负荷变化率 (%)' if HAS_CJK_FONT else 'Load Change (%)', fontsize=12)
            ax2.set_title('阶段间负荷变化率' if HAS_CJK_FONT else 'Load Change Rate', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. 特征重要性横向柱状图
        ax3 = fig.add_subplot(gs[1, 1])
        feat_imp = explanations.get('feature_importance', {})
        if feat_imp and feat_imp.get('correlations'):
            correlations = feat_imp['correlations']
            features = list(correlations.keys())[:5]  # 取前5个
            corr_values = [correlations[f] for f in features]
            
            colors = ['green' if c > 0 else 'red' for c in corr_values]
            bars = ax3.barh(features, corr_values, color=colors, alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_xlabel('相关系数' if HAS_CJK_FONT else 'Correlation', fontsize=12)
            ax3.set_ylabel('环境特征' if HAS_CJK_FONT else 'Features', fontsize=12)
            ax3.set_title('特征与负荷相关性' if HAS_CJK_FONT else 'Feature-Load Correlation', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for bar, val in zip(bars, corr_values):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{val:+.3f}',
                        ha='left' if width > 0 else 'right', 
                        va='center', fontsize=9)
        
        # 4. 环境因素影响雷达图
        ax4 = fig.add_subplot(gs[2, :], projection='polar')
        env_impact = explanations.get('environmental_impact', {})
        if env_impact:
            features = list(env_impact.keys())
            # 归一化range值用于雷达图
            ranges = [env_impact[f]['range'] for f in features]
            max_range = max(ranges) if ranges else 1
            normalized_ranges = [r / max_range for r in ranges]
            
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
            normalized_ranges += normalized_ranges[:1]  # 闭合图形
            angles += angles[:1]
            
            ax4.plot(angles, normalized_ranges, 'o-', linewidth=2, color='blue')
            ax4.fill(angles, normalized_ranges, alpha=0.25, color='blue')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(features, fontsize=10)
            ax4.set_ylim(0, 1)
            ax4.set_title('环境因素波动幅度' if HAS_CJK_FONT else 'Environmental Factors Variation', 
                         fontsize=12, fontweight='bold', pad=20)
            ax4.grid(True)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ 可解释性可视化图已保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 生成可视化失败: {e}")
        import traceback
        traceback.print_exc()

def simple_load_segmentation(load_values, n_segments=4, min_segment_length=8):
    """
    简单的负荷分段方法（作为HMM的备选方案）
    基于负荷水平的分位数进行分段，并合并短段
    """
    try:
        load_values = np.array(load_values)
        
        # 对数据进行轻微平滑，减少噪声
        from scipy import ndimage
        smoothed_values = ndimage.median_filter(load_values.astype(float), size=3)
        
        # 计算分位数阈值
        quantiles = np.linspace(0, 1, n_segments + 1)
        thresholds = np.quantile(smoothed_values, quantiles)
        
        # 分配状态
        raw_states = np.digitize(smoothed_values, thresholds[1:-1])
        
        # 应用滑动窗口平滑状态序列
        window_size = 5
        smoothed_states = np.zeros_like(raw_states)
        for i in range(len(raw_states)):
            start = max(0, i - window_size // 2)
            end = min(len(raw_states), i + window_size // 2 + 1)
            window = raw_states[start:end]
            # 使用众数作为平滑后的状态
            from scipy import stats
            smoothed_states[i] = int(stats.mode(window)[0])
        
        # 计算每个状态的平均值
        state_means = []
        for state in range(n_segments):
            state_mask = (smoothed_states == state)
            if np.any(state_mask):
                state_mean = np.mean(load_values[state_mask])
                state_means.append(state_mean)
            else:
                state_means.append(np.mean(load_values))
        
        state_means = np.array(state_means)
        
        # 识别初始连续段
        initial_segments = []
        current_state = smoothed_states[0]
        start_idx = 0
        
        for i in range(1, len(smoothed_states)):
            if smoothed_states[i] != current_state:
                end_idx = i - 1
                segment_load = np.mean(load_values[start_idx:i])
                initial_segments.append((start_idx, end_idx, current_state, segment_load))
                start_idx = i
                current_state = smoothed_states[i]
        
        # 添加最后一段
        segment_load = np.mean(load_values[start_idx:])
        initial_segments.append((start_idx, len(smoothed_states) - 1, current_state, segment_load))
        
        # 合并短段
        merged_segments = merge_short_segments(initial_segments, load_values, min_segment_length)
        
        # 重新构建状态序列
        final_states = np.zeros_like(smoothed_states)
        for start_idx, end_idx, state, _ in merged_segments:
            final_states[start_idx:end_idx+1] = state
        
        # 重新计算状态平均值
        final_state_means = []
        unique_states = sorted(set([seg[2] for seg in merged_segments]))
        for state in unique_states:
            state_mask = (final_states == state)
            if np.any(state_mask):
                state_mean = np.mean(load_values[state_mask])
                final_state_means.append(state_mean)
            else:
                final_state_means.append(np.mean(load_values))
        
        final_state_means = np.array(final_state_means)
        
        return final_states, final_state_means, merged_segments
        
    except Exception as e:
        print(f"❌ 简单分段也失败: {e}")
        # 最简单的退回方案
        n = len(load_values)
        states = np.zeros(n, dtype=int)
        state_means = np.array([np.mean(load_values)])
        segments = [(0, n-1, 0, np.mean(load_values))]
        return states, state_means, segments


def plot_prediction_day(ts, dates_val, y_val, y_pred_val, out_dir, plot_date=None, step_minutes=15):
    """绘制并保存某一天的预测对比图。
    """
    try:
        if not dates_val:
            print("无验证样本，跳过绘图。")
            return

        if plot_date is None:
            selected_date = (pd.Timestamp(dates_val[0]) + pd.Timedelta(minutes=step_minutes)).date()
        else:
            selected_date = pd.to_datetime(plot_date).date()

        # 根据是否有中文字体，选择标签文本
        if HAS_CJK_FONT:
            label_true = '真实（目标下一时刻）'
            label_pred = '预测'
            label_smooth = '原始平滑（日曲线）'
            xlabel = '时间步（索引）'
            ylabel = '负荷'
            title_fmt = '负荷预测对比 - {}'
        else:
            label_true = 'True (target next)'
            label_pred = 'Prediction'
            label_smooth = 'Original smooth (day)'
            xlabel = 'Time step (index)'
            ylabel = 'Load'
            title_fmt = 'Load prediction comparison - {}'

        # 构建验证集中时间->预测/真实映射
        pred_map = {}
        true_map = {}
        for dt, true_v, pred_v in zip(dates_val, y_val, y_pred_val):
            target_time = pd.Timestamp(dt) + pd.Timedelta(minutes=step_minutes)
            key = target_time.floor('min')
            true_map[key] = true_v
            pred_map[key] = pred_v

        day_len = int(24 * 60 / step_minutes)
        base = pd.Timestamp(selected_date)
        day_index = pd.date_range(start=base, periods=day_len, freq=f"{step_minutes}T")

        true_series = pd.Series(index=list(true_map.keys()), data=list(true_map.values()))
        pred_series = pd.Series(index=list(pred_map.keys()), data=list(pred_map.values()))

        true_series = true_series[true_series.index.date == selected_date]
        pred_series = pred_series[pred_series.index.date == selected_date]

        true_full = true_series.reindex(day_index).interpolate(method='time').ffill().bfill()
        pred_full = pred_series.reindex(day_index).interpolate(method='time').ffill().bfill()

        y_true_day_full = true_full.values
        y_pred_day_full = pred_full.values

        # 原始当天负荷
        day_series = ts[ts.index.date == selected_date]
        if not day_series.empty:
            raw_vals = day_series['load'].values
            if len(raw_vals) != day_len:
                tmp = day_series['load'].reindex(day_index, method='nearest')
                raw_vals = tmp.values
            smooth_vals = gaussian_filter1d(raw_vals.astype(float), sigma=2.0)
        else:
            smooth_vals = None

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        time_axis = np.arange(day_len)
        ax.plot(time_axis, y_true_day_full, label=label_true, marker='o')
        ax.plot(time_axis, y_pred_day_full, label=label_pred, marker='x')

        if smooth_vals is not None and len(smooth_vals) == day_len:
            ax.plot(time_axis, smooth_vals, label=label_smooth, alpha=0.6)

        ax.set_title(title_fmt.format(selected_date.strftime('%Y-%m-%d')))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.tight_layout()

        out_fig = os.path.join(out_dir, 'prediction_vs_true_{}.png'.format(selected_date.strftime('%Y%m%d')))
        fig.savefig(out_fig, dpi=200)
        plt.close(fig)
        print('已保存预测对比图: {}'.format(out_fig))
    except Exception as e:
        print('绘图出错: {}'.format(e))


def plot_all_valid_days(ts, dates_val, y_val, y_pred_val, out_dir, step_minutes=15):
    """为验证集中每个出现目标时间的日期生成日负荷对比图并保存。
    仅对验证集中实际出现的日期绘图（这些日期对应已有完整历史窗口的样本）。
    """
    try:
        if not dates_val:
            print('无验证样本，跳过批量日绘图。')
            return
        # 目标时间 = 当前样本时间 + step_minutes
        target_times = [pd.Timestamp(dt) + pd.Timedelta(minutes=step_minutes) for dt in dates_val]
        target_dates = pd.to_datetime(target_times).date
        unique_dates = sorted(set(target_dates))
        for ud in unique_dates:
            # 收集当天的索引和值
            idxs = [i for i, tt in enumerate(target_times) if tt.date() == ud]
            if not idxs:
                continue
            # 为该天生成子数组并调用绘图
            sub_dates = [dates_val[i] for i in idxs]
            sub_y = [y_val[i] for i in idxs]
            sub_pred = [y_pred_val[i] for i in idxs]
            try:
                plot_prediction_day(ts, sub_dates, sub_y, sub_pred, out_dir, plot_date=str(ud), step_minutes=step_minutes)
            except Exception as e:
                print('绘图失败 {}: {}'.format(ud, e))
    except Exception as e:
        print('批量日绘图出错: {}'.format(e))


def visualize_results(ts, dates_val, y_val, y_pred_val, out_dir, plot_date=None, step_minutes=15):
    """生成并保存多种模型效果图：
    - 单日对比（使用已有 plot_prediction_day）
    - 验证集散点图（真实 vs 预测）
    - 残差直方图
    - 残差随时间序列图
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style='whitegrid')

        # 转换为numpy
        y_true = np.array(y_val)
        y_pred = np.array(y_pred_val)
        resid = y_true - y_pred

        # 1) 单日对比图
        try:
            plot_prediction_day(ts, dates_val, y_val, y_pred_val, out_dir, plot_date=plot_date, step_minutes=step_minutes)
        except Exception as e:
            print('单日对比绘图失败: {}'.format(e))

        # 2) 散点图：真实 vs 预测
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_true, y_pred, alpha=0.6, s=10)
            mn = min(y_true.min(), y_pred.min())
            mx = max(y_true.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], 'r--')
            if HAS_CJK_FONT:
                ax.set_xlabel('真实')
                ax.set_ylabel('预测')
                ax.set_title('真实 vs 预测 (验证集)')
            else:
                ax.set_xlabel('True')
                ax.set_ylabel('Pred')
                ax.set_title('True vs Pred (val)')
            fig.savefig(os.path.join(out_dir, 'scatter_true_vs_pred_val.png'), dpi=200)
            plt.close(fig)
        except Exception as e:
            print('散点图绘制失败: {}'.format(e))

        # 3) 残差直方图
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(resid, bins=50, kde=True, ax=ax)
            if HAS_CJK_FONT:
                ax.set_title('残差分布 (真实 - 预测)')
                ax.set_xlabel('残差')
            else:
                ax.set_title('Residual distribution (true - pred)')
                ax.set_xlabel('Residual')
            fig.savefig(os.path.join(out_dir, 'residual_hist_val.png'), dpi=200)
            plt.close(fig)
        except Exception as e:
            print('残差直方图绘制失败: {}'.format(e))

        # 4) 残差随时间图
        try:
            if dates_val:
                times = pd.to_datetime(dates_val)
                df_r = pd.DataFrame({'time': times, 'resid': resid, 'true': y_true, 'pred': y_pred})
                df_r = df_r.set_index('time')
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(df_r.index, df_r['resid'], marker='.', markersize=3, linestyle='-')
                if HAS_CJK_FONT:
                    ax.set_title('残差随时间 (验证集)')
                    ax.set_xlabel('时间')
                    ax.set_ylabel('残差')
                else:
                    ax.set_title('Residuals over time (val)')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Residual')
                fig.savefig(os.path.join(out_dir, 'residual_time_series_val.png'), dpi=200)
                plt.close(fig)
        except Exception as e:
            print('残差随时间绘图失败: {}'.format(e))



        print('可视化图像已保存到: {}'.format(out_dir))
    except Exception as e:
        print('可视化生成失败: {}'.format(e))


def select_mode():
    """选择运行模式：训练或预测"""
    print("\n🏠 家庭负荷预测系统")
    print("="*60)
    print("请选择运行模式：")
    print("1. 训练模式 - 训练新的LSTM模型")
    print("2. 预测模式 - 使用已保存的模型进行预测")
    
    while True:
        try:
            choice = input("\n请输入选择 (1/2): ").strip()
            if choice == '1':
                return 'train'
            elif choice == '2':
                return 'predict'
            else:
                print("❌ 请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消操作")
            return 'cancel'

def train_mode():
    """训练模式"""
    # 交互式选择要训练的户数
    result = interactive_select_households()
    if len(result) == 3:
        selected_files, mode, custom_model_name = result
    else:
        selected_files, mode = result
        custom_model_name = None
    
    if not selected_files or mode in ['cancel', 'error']:
        print("❌ 未选择任何文件或操作取消")
        return
    
    print(f"\n🚀 开始单户LSTM训练")
    print("="*60)
    
    # 单户详细分析模式
    csv_path = selected_files[0]
    household_name = extract_household_name(csv_path)
    
    # 使用自定义模型名称或默认家庭名称
    model_name = custom_model_name if custom_model_name else household_name
    
    # 为当前模型创建专用的保存目录
    model_save_dir = os.path.join(MODEL_OUTPUT_DIR, model_name)
    analysis_save_dir = os.path.join(ANALYSIS_OUTPUT_DIR, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(analysis_save_dir, exist_ok=True)
    
    print('单户详细分析模式，使用文件: {}'.format(os.path.basename(csv_path)))
    print('家庭数据: {}'.format(household_name))
    print('模型名称: {}'.format(model_name))
    print('模型保存目录: {}'.format(model_save_dir))
    print('分析结果保存目录: {}'.format(analysis_save_dir))
    
    ts = load_time_series(csv_path)
    if ts.empty:
        print('加载时间序列失败或无数据')
        return

    feat_df = build_time_features(ts, sigma=2.0, seq_len=SEQ_LEN)
    if feat_df.empty:
        print('特征提取失败')
        return

    feat_df = feat_df.fillna(method='ffill').fillna(method='bfill')
    for c in feat_df.columns:
        if feat_df[c].isna().any():
            med = feat_df[c].median()
            feat_df[c].fillna(med if not np.isnan(med) else 0.0, inplace=True)

    X, y, dates = build_sequences_from_features(feat_df, seq_days=1, step_per_day=96)
    valid_idx = [i for i in range(len(y)) if (not np.isnan(y[i])) and (not np.isnan(X[i]).any())]
    X = X[valid_idx]; y = y[valid_idx]; dates = [dates[i] for i in valid_idx]

    (X_train, y_train, d_train), (X_test, y_test, d_test), (X_val, y_val, d_val) = time_order_split(X, y, dates, train_frac=0.7, test_frac=0.15)
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    X_train_s = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)

    model = build_lstm_model(input_shape=(X_train_s.shape[1], X_train_s.shape[2]))
    es = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    
    print(f"\n📊 数据划分信息：")
    print(f"   • 训练集：{len(X_train)} 样本")
    print(f"   • 测试集：{len(X_test)} 样本") 
    print(f"   • 验证集：{len(X_val)} 样本")
    print(f"   • 特征维度：{n_features}")
    
    print(f"\n🔥 开始训练模型...")
    model.fit(X_train_s, y_train, validation_data=(X_test_s, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)

    y_pred_val = model.predict(X_val_s).reshape(-1)
    y_pred_test = model.predict(X_test_s).reshape(-1)

    # 估计不确定度
    train_pred = model.predict(X_train_s).reshape(-1)
    resid = y_train - train_pred
    sigma_global = float(np.std(resid)) if len(resid) > 0 else 1e-6

    # 计算所有指标
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100.0
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    val_mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100.0 if len(y_val) > 0 else 0.0
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val)) if len(y_val) > 0 else 0.0
    val_r2 = r2_score(y_val, y_pred_val) if len(y_val) > 0 else 0.0

    # 打印指标
    print(f"\n📊 模型性能评估：")
    print(f"   测试集 MAPE: {test_mape:.2f}%")
    print(f"   测试集 RMSE: {test_rmse:.4f}")
    print(f"   测试集 R²:   {test_r2:.4f}")
    if len(y_val) > 0:
        print(f"   验证集 MAPE: {val_mape:.2f}%")
        print(f"   验证集 RMSE: {val_rmse:.4f}")
        print(f"   验证集 R²:   {val_r2:.4f}")

    # 使用模型专用输出目录
    out_dir = analysis_save_dir
    
    # 保存模型到专用模型目录
    try:
        model_path = os.path.join(model_save_dir, f'{model_name}_lstm_model.h5')
        model.save(model_path)
        print(f"✅ 模型已保存到: {model_path}")
        
        # 保存scaler以便预测时使用
        import pickle
        scaler_path = os.path.join(model_save_dir, f'{model_name}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ 标准化器已保存到: {scaler_path}")
        
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")

    # 保存训练指标
    try:
        results_single = {
            'model_name': model_name,
            'household_name': household_name,
            'file': os.path.basename(csv_path),
            'test_mape_pct': float(test_mape),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2),
            'val_mape_pct': float(val_mape),
            'val_rmse': float(val_rmse),
            'val_r2': float(val_r2),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'train_samples': len(y_train),
            'n_features': n_features,
            'mode': mode,
            'sigma_global': float(sigma_global)
        }
        metrics_path = os.path.join(out_dir, f'{model_name}_metrics.csv')
        pd.DataFrame([results_single]).to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"✅ 训练指标已保存到: {metrics_path}")
    except Exception as e:
        print(f"❌ 训练指标保存失败: {e}")
    
    print(f"\n🎉 单户训练完成！")
    print(f"   模型名称: {model_name}")
    print(f"   家庭数据: {household_name}")
    print(f"   文件: {os.path.basename(csv_path)}")
    print(f"   测试MAPE: {test_mape:.2f}%")
    print(f"   模型保存在: {model_save_dir}")
    print(f"   分析结果保存在: {analysis_save_dir}")
    return

def select_saved_model():
    """选择已保存的模型"""
    print("\n🔍 查找已保存的模型...")
    
    # 查找所有已保存的模型
    model_files = []
    for root, dirs, files in os.walk(MODEL_OUTPUT_DIR):
        for file in files:
            if file.endswith('_lstm_model.h5'):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        print("❌ 未找到任何已保存的模型文件")
        return None, None, None
    
    print(f"📋 发现 {len(model_files)} 个已保存的模型：")
    for i, model_file in enumerate(model_files, 1):
        # 提取模型名称
        basename = os.path.basename(model_file)
        model_name = basename.replace('_lstm_model.h5', '')
        model_dir = os.path.dirname(model_file)
        parent_dir = os.path.basename(model_dir)
        print(f"  {i:2d}. {model_name} (目录: {parent_dir})")
    
    while True:
        try:
            choice = input(f"\n请选择模型 (1-{len(model_files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(model_files):
                model_path = model_files[idx]
                basename = os.path.basename(model_path)
                model_name = basename.replace('_lstm_model.h5', '')
                
                # 查找对应的scaler文件
                model_dir = os.path.dirname(model_path)
                scaler_path = os.path.join(model_dir, f'{model_name}_scaler.pkl')
                
                if not os.path.exists(scaler_path):
                    print(f"❌ 未找到对应的标准化器文件: {scaler_path}")
                    continue
                
                # 需要用户指定对应的数据文件，因为模型名称可能与数据文件名不匹配
                print(f"\n选择的模型: {model_name}")
                print("请选择对应的数据文件...")
                
                # 查找所有可用的数据文件
                all_data_files = find_apartment_files(DATA_FOLDER)
                print("📋 可用数据文件：")
                for i, data_file in enumerate(all_data_files[:10], 1):  # 只显示前10个
                    basename = os.path.basename(data_file)
                    print(f"  {i:2d}. {basename}")
                if len(all_data_files) > 10:
                    print(f"  ... 还有 {len(all_data_files)-10} 个文件")
                
                while True:
                    try:
                        data_choice = input(f"\n请选择数据文件序号 (1-{len(all_data_files)}): ").strip()
                        data_idx = int(data_choice) - 1
                        if 0 <= data_idx < len(all_data_files):
                            data_file = all_data_files[data_idx]
                            print(f"✅ 选择模型: {model_name}")
                            print(f"✅ 选择数据: {os.path.basename(data_file)}")
                            return model_path, scaler_path, data_file
                        else:
                            print(f"❌ 请输入 1-{len(all_data_files)} 之间的数字")
                    except ValueError:
                        print("❌ 请输入有效的数字")
                    except KeyboardInterrupt:
                        print("\n❌ 用户取消操作")
                        break
            else:
                print(f"❌ 请输入 1-{len(model_files)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消操作")
            return None, None, None

def select_prediction_date(ts):
    """选择要预测的日期"""
    print(f"\n📅 数据时间范围：")
    print(f"   开始时间: {ts.index[0]}")
    print(f"   结束时间: {ts.index[-1]}")
    
    # 计算可用的预测日期范围（需要足够的历史数据）
    seq_len = SEQ_LEN * 96  # 14天 * 96个点/天
    available_start = ts.index[seq_len] if len(ts.index) > seq_len else ts.index[0]
    available_end = ts.index[-96] if len(ts.index) > 96 else ts.index[-1]
    
    print(f"   可预测范围: {available_start.date()} 至 {available_end.date()}")
    
    while True:
        try:
            date_str = input("\n请输入要预测的日期 (YYYY-MM-DD): ").strip()
            target_date = pd.to_datetime(date_str)
            
            if target_date < available_start:
                print(f"❌ 预测日期太早，需要足够的历史数据（至少{SEQ_LEN}天）")
                continue
            if target_date > available_end:
                print(f"❌ 预测日期太晚，没有足够的数据用于预测")
                continue
                
            # 找到最接近的时间点
            closest_idx = ts.index.get_indexer([target_date], method='nearest')[0]
            actual_date = ts.index[closest_idx]
            
            print(f"✅ 预测日期: {actual_date.date()}")
            return actual_date
            
        except ValueError:
            print("❌ 日期格式错误，请使用 YYYY-MM-DD 格式")
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消操作")
            return None

def predict_single_day(model, scaler, feat_df, target_date, step_per_day=96):
    """预测单天的负荷 - 预测目标日期一整天96个时间点的负荷值"""
    # 注意：这里的序列长度应该与训练时一致
    # 训练时使用的是 seq_days=1, step_per_day=96，所以序列长度是96
    seq_len = step_per_day  # 96个时间步，对应1天的96个15分钟间隔
    
    # 找到目标日期的开始索引
    target_date_start = pd.Timestamp(target_date.date())
    
    # 生成目标日期的所有时间点（96个15分钟间隔）
    target_times = pd.date_range(start=target_date_start, periods=step_per_day, freq='15T')
    
    predictions = []
    actual_times = []
    
    # 提取序列特征（排除target列）
    feat_cols = [c for c in feat_df.columns if c != 'target_next']
    n_features = len(feat_cols)
    
    for i, target_time in enumerate(target_times):
        try:
            # 找到目标时间在特征数据中的索引
            target_idx = feat_df.index.get_indexer([target_time], method='nearest')[0]
            
            # 检查是否有足够的历史数据
            if target_idx < seq_len:
                continue
            
            # 提取序列特征
            seq_data = feat_df.iloc[target_idx-seq_len:target_idx][feat_cols].values
            
            # 检查序列数据的形状
            if seq_data.shape[0] != seq_len:
                continue
            
            # 重塑数据为正确的形状：(seq_len, n_features)
            seq_reshaped = seq_data.reshape(seq_len, n_features)
            
            # 标准化 - 逐个时间步标准化
            seq_scaled = scaler.transform(seq_reshaped)
            
            # 重塑为模型输入格式：(batch_size=1, seq_len, n_features)
            seq_scaled = seq_scaled.reshape(1, seq_len, n_features)
            
            # 预测
            prediction = model.predict(seq_scaled, verbose=0)[0, 0]
            
            predictions.append(float(prediction))
            actual_times.append(target_time)
            
        except Exception as e:
            print(f"预测时间点 {target_time} 失败: {e}")
            continue
    
    return predictions, actual_times

def plot_single_day_prediction(ts, feat_df, pred_date, pred_values, pred_times, out_dir, step_minutes=15):
    """绘制单天预测结果，包含HMM智能阶段划分"""
    try:
        day_len = int(24 * 60 / step_minutes)
        pred_date_obj = pred_date.date()
        
        # 获取当天的实际数据
        day_series = ts[ts.index.date == pred_date_obj]
        if day_series.empty:
            print(f"❌ 未找到日期 {pred_date_obj} 的实际数据")
            return None
        
        # 生成当天的时间索引
        base = pd.Timestamp(pred_date_obj)
        day_index = pd.date_range(start=base, periods=day_len, freq=f"{step_minutes}T")
        
        # 重采样实际数据到标准时间点
        if len(day_series) != day_len:
            day_series_resampled = day_series['load'].reindex(day_index, method='nearest')
        else:
            day_series_resampled = day_series['load']
        
        # 平滑处理
        smooth_vals = gaussian_filter1d(day_series_resampled.values.astype(float), sigma=2.0)
        
        # 处理预测数据 - 将预测值对齐到标准时间点
        pred_series = pd.Series(index=pred_times, data=pred_values)
        pred_resampled = pred_series.reindex(day_index, method='nearest').fillna(method='ffill').fillna(method='bfill')
        
        # 使用HMM对预测负荷进行智能阶段划分（优化版本，减少过度分段）
        print("🔄 正在进行HMM负荷阶段划分...")
        try:
            # 动态计算最小段长度：一天分为3-8段，每段至少1.5-2小时
            min_segment_length = max(6, len(pred_resampled) // 8)  # 至少6个点（1.5小时）
            states, state_means, segments = hmm_load_segmentation(
                pred_resampled.values, 
                n_states='auto', 
                min_states=3, 
                max_states=5,  # 减少最大状态数
                min_segment_length=min_segment_length
            )
            print(f"✅ HMM划分完成：识别出 {len(segments)} 个负荷阶段")
            
            # 打印阶段信息
            for i, (start_idx, end_idx, state, mean_load) in enumerate(segments):
                start_time = start_idx * step_minutes / 60
                end_time = (end_idx + 1) * step_minutes / 60
                print(f"   阶段 {i+1}: {start_time:05.2f}h-{end_time:05.2f}h, 状态={state}, 平均负荷={mean_load:.3f}")
        except Exception as e:
            print(f"⚠️ HMM划分失败，使用简单分段: {e}")
            states, state_means, segments = simple_load_segmentation(pred_resampled.values, n_segments=4)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(14, 8))
        time_axis = np.arange(day_len)
        
        # 转换时间轴为小时
        time_axis_hours = time_axis * step_minutes / 60  # 转换为小时
        
        # 定义颜色方案 - 为不同阶段使用不同颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        
        # 绘制阶段背景填充
        for i, (start_idx, end_idx, state, mean_load) in enumerate(segments):
            start_hour = start_idx * step_minutes / 60
            end_hour = (end_idx + 1) * step_minutes / 60
            
            # 填充背景区域
            ax.axvspan(start_hour, end_hour, alpha=0.3, color=colors[i], 
                      label=f'阶段{i+1} (状态{state})' if HAS_CJK_FONT else f'Stage{i+1} (State{state})')
            
            # 绘制阶段平均负荷的水平线
            ax.hlines(mean_load, start_hour, end_hour, colors=colors[i], 
                     linestyles='--', linewidth=3, alpha=0.8)
            
            # 添加阶段标注
            mid_hour = (start_hour + end_hour) / 2
            ax.text(mid_hour, mean_load + 0.1, f'阶段{i+1}\n{mean_load:.2f}' if HAS_CJK_FONT else f'Stage{i+1}\n{mean_load:.2f}', 
                   ha='center', va='bottom', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
        
        # 绘制实际负荷曲线
        ax.plot(time_axis_hours, smooth_vals, label='实际负荷' if HAS_CJK_FONT else 'Actual Load', 
                color='blue', linewidth=2.5, alpha=0.9, marker='o', markersize=1, zorder=3)
        
        # 绘制预测负荷曲线
        ax.plot(time_axis_hours, pred_resampled.values, label='预测负荷' if HAS_CJK_FONT else 'Predicted Load',
                color='red', linewidth=2.5, alpha=0.9, marker='x', markersize=1, zorder=3)
        
        # 设置坐标轴
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_ylim(0, 4)
        ax.set_yticks(np.arange(0, 4.5, 0.5))
        
        # 设置标签和标题
        if HAS_CJK_FONT:
            ax.set_title(f'智能负荷预测与阶段划分 - {pred_date_obj}', fontsize=14, fontweight='bold')
            ax.set_xlabel('时间 (小时)', fontsize=12)
            ax.set_ylabel('负荷', fontsize=12)
        else:
            ax.set_title(f'Smart Load Prediction & Stage Segmentation - {pred_date_obj}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (Hours)', fontsize=12)
            ax.set_ylabel('Load', fontsize=12)
        
        # 设置图例
        handles, labels = ax.get_legend_handles_labels()
        # 将阶段图例和曲线图例分开
        stage_handles = handles[:-2]  # 阶段填充
        curve_handles = handles[-2:]  # 曲线
        
        # 创建两个图例
        legend1 = ax.legend(curve_handles, labels[-2:], loc='upper left', fontsize=10)
        ax.add_artist(legend1)  # 添加第一个图例
        
        if stage_handles:  # 如果有阶段
            legend2 = ax.legend(stage_handles, labels[:-2], loc='upper right', fontsize=8, ncol=2)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        out_file = os.path.join(out_dir, f'prediction_with_stages_{pred_date_obj.strftime("%Y%m%d")}.png')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # 计算当天的统计信息
        actual_mean = np.mean(smooth_vals)
        pred_mean = np.mean(pred_resampled.values)
        error = abs(pred_mean - actual_mean)
        mape = (error / actual_mean) * 100 if actual_mean != 0 else 0
        
        print(f"✅ 已保存阶段划分预测图: {out_file}")
        print(f"   实际日均负荷: {actual_mean:.4f}")
        print(f"   预测日均负荷: {pred_mean:.4f}")
        print(f"   绝对误差: {error:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        # 生成负荷变化可解释性分析
        print("\n🔍 生成负荷变化可解释性分析...")
        explanations = explain_load_changes(segments, feat_df, pred_times, pred_resampled.values)
        
        # 保存可解释性报告
        report_path = os.path.join(out_dir, f'explanation_report_{pred_date_obj.strftime("%Y%m%d")}.txt')
        generate_explanation_report(explanations, report_path)
        
        # 保存可解释性可视化
        viz_path = os.path.join(out_dir, f'explanation_viz_{pred_date_obj.strftime("%Y%m%d")}.png')
        visualize_explanations(explanations, viz_path)
        
        # 打印简要解释
        print("\n📊 负荷变化解释摘要:")
        for seg in explanations.get('segment_analysis', []):
            print(f"   阶段{seg['segment_id']}: {seg['load_level']} ({seg['start_time']:.1f}h-{seg['end_time']:.1f}h)")
            if seg['key_factors']:
                print(f"      关键因素: {seg['key_factors'][0]}")
        
        return {
            'date': pred_date_obj,
            'actual_mean': actual_mean,
            'predicted_mean': pred_mean,
            'error': error,
            'mape': mape,
            'image_path': out_file,
            'predictions': pred_values,
            'pred_times': pred_times,
            'segments': segments,
            'states': states.tolist(),
            'state_means': state_means.tolist(),
            'explanations': explanations,
            'explanation_report': report_path,
            'explanation_viz': viz_path
        }
        
    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_mode():
    """预测模式"""
    print(f"\n🔮 预测模式")
    print("="*60)

    # 选择模型
    model_path, scaler_path, data_path = select_saved_model()
    if not model_path:
        return

    household_name = extract_household_name(data_path)
    # 从模型路径提取模型名称
    model_basename = os.path.basename(model_path)
    model_name = model_basename.replace('_lstm_model.h5', '')

    print(f"📂 使用模型: {model_name}")
    print(f"📂 使用数据文件: {os.path.basename(data_path)}")

    # 加载模型和标准化器
    try:
        import pickle
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ 模型和标准化器加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 加载和预处理数据
    print(f"\n📊 处理数据...")
    ts = load_time_series(data_path)
    if ts.empty:
        print('❌ 加载时间序列失败或无数据')
        return

    feat_df = build_time_features(ts, sigma=2.0, seq_len=SEQ_LEN)
    if feat_df.empty:
        print('❌ 特征提取失败')
        return

    feat_df = feat_df.fillna(method='ffill').fillna(method='bfill')
    for c in feat_df.columns:
        if feat_df[c].isna().any():
            med = feat_df[c].median()
            feat_df[c].fillna(med if not np.isnan(med) else 0.0, inplace=True)

    # 创建预测输出目录（使用模型名称）
    prediction_dir = os.path.join(ANALYSIS_OUTPUT_DIR, model_name, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    print(f"✅ 预测环境准备完成")
    print(f"   预测结果保存到: {prediction_dir}")

    # 进入预测循环
    while True:
        try:
            print(f"\n{'='*60}")
            print(f"🔮 预测模式 - 使用模型: {model_name}")
            print(f"{'='*60}")

            # 选择预测日期
            target_datetime = select_prediction_date(ts)
            if not target_datetime:
                print("❌ 未选择有效日期")
                continue

            print(f"\n🎯 开始预测指定日期...")
            print(f"   预测日期: {target_datetime.date()}")

            # 单日预测
            step_per_day = 96

            try:
                # 预测整天的负荷值
                pred_values, pred_times = predict_single_day(model, scaler, feat_df, target_datetime, step_per_day)

                if not pred_values:
                    print(f"❌ 无法为日期 {target_datetime.date()} 生成预测值")
                    continue

                print(f"✅ 成功预测 {len(pred_values)} 个时间点的负荷值")

                # 绘图并保存
                result = plot_single_day_prediction(ts, feat_df, target_datetime, pred_values, pred_times, prediction_dir)

                if result:
                    # 保存详细预测结果
                    detailed_results = []
                    for i, (pred_val, pred_time) in enumerate(zip(pred_values, pred_times)):
                        detailed_results.append({
                            'date': target_datetime.date(),
                            'time': pred_time.strftime('%H:%M:%S'),
                            'predicted_load': pred_val,
                            'time_index': i
                        })

                    # 保存详细预测数据
                    detail_path = os.path.join(prediction_dir, f"prediction_detail_{target_datetime.date().strftime('%Y%m%d')}.csv")
                    pd.DataFrame(detailed_results).to_csv(detail_path, index=False, encoding='utf-8-sig')

                    # 保存汇总结果（排除不能序列化的对象）
                    result_for_save = {
                        'date': result['date'],
                        'actual_mean': result['actual_mean'],
                        'predicted_mean': result['predicted_mean'],
                        'error': result['error'],
                        'mape': result['mape'],
                        'image_path': result['image_path'],
                        'num_segments': len(result.get('segments', [])),
                        'explanation_report': result.get('explanation_report', ''),
                        'explanation_viz': result.get('explanation_viz', '')
                    }
                    result_path = os.path.join(prediction_dir, f"prediction_summary_{target_datetime.date().strftime('%Y%m%d')}.csv")
                    pd.DataFrame([result_for_save]).to_csv(result_path, index=False, encoding='utf-8-sig')

                    print(f"✅ 详细预测结果已保存到: {detail_path}")
                    print(f"✅ 汇总预测结果已保存到: {result_path}")
                    
                    # 保存可解释性分析结果（JSON格式）
                    if 'explanations' in result:
                        import json
                        explanation_json_path = os.path.join(prediction_dir, f"explanation_{target_datetime.date().strftime('%Y%m%d')}.json")
                        with open(explanation_json_path, 'w', encoding='utf-8') as f:
                            json.dump(result['explanations'], f, ensure_ascii=False, indent=2)
                        print(f"✅ 可解释性分析(JSON)已保存到: {explanation_json_path}")

                print(f"\n🎉 预测完成！")

            except Exception as e:
                print(f"❌ 预测日期 {target_datetime.date()} 失败: {e}")
                import traceback
                traceback.print_exc()

            # 询问是否继续预测
            print(f"\n{'='*50}")
            while True:
                try:
                    choice = input("是否继续预测其他日期？(y/n): ").strip().lower()
                    if choice in ['y', 'yes', '是', '']:
                        break  # 继续外层循环
                    elif choice in ['n', 'no', '否']:
                        print(f"\n🎉 预测模式结束！感谢使用！")
                        return
                    else:
                        print("❌ 请输入 y 或 n")
                except KeyboardInterrupt:
                    print(f"\n\n🎉 预测模式结束！感谢使用！")
                    return

        except KeyboardInterrupt:
            print(f"\n\n🎉 预测模式结束！感谢使用！")
            return
        except Exception as e:
            print(f"❌ 预测过程出错: {e}")
            # 继续循环，不退出程序

def main():
    """主函数"""
    mode = select_mode()

    if mode == 'train':
        train_mode()
    elif mode == 'predict':
        predict_mode()
    elif mode == 'cancel':
        print("❌ 操作已取消")
    else:
        print("❌ 未知模式")

if __name__ == '__main__':
    main()

