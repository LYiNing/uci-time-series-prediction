# 导入常用的科学计算与可视化库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 导入处理时间序列所需的库
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# 导入时间序列分析相关模型
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
# 导入 PyTorch 深度学习相关组件
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')
# 设置随机种子，确保实验可复现
np.random.seed(3407)
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)


class DFT_series_decomp(nn.Module):
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=-1)
        freq = torch.abs(xf)
        freq[:, :, 0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k, dim=-1)

        mask = torch.zeros_like(xf, dtype=torch.bool)
        mask.scatter_(-1, top_list, True)
        xf[~mask] = 0

        x_season = torch.fft.irfft(xf, n=x.size(-1))
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.num_scales - 1)
            ]
        )

    def forward(self, season_list):
        batch_size, num_features, _ = season_list[0].shape
        # 【修正】: 使用 .reshape() 代替 .view()
        season_list_flat = [s.reshape(batch_size * num_features, -1) for s in season_list]

        out_high = season_list_flat[0]
        out_season_list_flat = [out_high]

        for i in range(len(season_list_flat) - 1):
            out_low = season_list_flat[i + 1]
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            out_season_list_flat.append(out_high)

        # 【修正】: 使用 .reshape() 代替 .view()
        out_season_list = [s.reshape(batch_size, num_features, -1) for s in out_season_list_flat]
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.num_scales - 1))
            ])

    def forward(self, trend_list):
        batch_size, num_features, _ = trend_list[0].shape
        # 【修正】: 使用 .reshape() 代替 .view()
        trend_list_flat = [t.reshape(batch_size * num_features, -1) for t in trend_list]

        trend_list_reverse = trend_list_flat.copy()
        trend_list_reverse.reverse()

        out_low = trend_list_reverse[0]
        out_trend_list_flat = [out_low]

        for i in range(len(trend_list_reverse) - 1):
            out_high = trend_list_reverse[i + 1]
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            out_trend_list_flat.append(out_low)

        out_trend_list_flat.reverse()
        # 【修正】: 使用 .reshape() 代替 .view()
        out_trend_list = [t.reshape(batch_size, num_features, -1) for t in out_trend_list_flat]
        return out_trend_list


class TimeMixer(nn.Module):
    """
    集成了多尺度分解与混合的 Transformer。
    """

    def __init__(self, input_size, output_size=90, d_model=64, nhead=8, num_layers=3, dropout=0.1,
                 num_scales=3, top_k=5, max_seq_len=1000):
        super(TimeMixer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model

        self.num_scales = num_scales
        self.top_k = top_k

        self.decomp_modules = None
        self.season_mixer = None
        self.trend_mixer = None

        self.input_projection = nn.Linear(input_size * 2, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def _initialize_mixers(self, seq_len, device):
        """内部函数，用于在第一次前向传播时创建混合模块"""

        class Config:
            pass

        configs = Config()
        configs.seq_len = seq_len
        configs.num_scales = self.num_scales
        configs.down_sampling_window = 1

        self.decomp_modules = nn.ModuleList([DFT_series_decomp(top_k=self.top_k) for _ in range(self.num_scales)]).to(
            device)
        self.season_mixer = MultiScaleSeasonMixing(configs).to(device)
        self.trend_mixer = MultiScaleTrendMixing(configs).to(device)

    def forward(self, x):
        """
        前向传播
        - x: 输入时间序列，形状为 [batch_size, seq_len, input_size]
        """
        batch_size, seq_len, _ = x.shape

        if self.season_mixer is None:
            self._initialize_mixers(seq_len, x.device)

        x_permuted = x.permute(0, 2, 1)  # -> [B, F, L]
        season_list, trend_list = [], []
        residual_trend = x_permuted
        for i in range(self.num_scales):
            season_part, trend_part = self.decomp_modules[i](residual_trend)
            season_list.append(season_part)
            trend_list.append(trend_part)
            residual_trend = trend_part

        mixed_season_list = self.season_mixer(season_list)
        mixed_trend_list = self.trend_mixer(trend_list)

        final_season_repr = sum(mixed_season_list).permute(0, 2, 1)
        final_trend_repr = sum(mixed_trend_list).permute(0, 2, 1)
        combined_repr = torch.cat([final_season_repr, final_trend_repr], dim=-1)

        x = self.input_projection(combined_repr)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)

        x = x[:, -1, :]
        x = self.dropout(x)
        output = self.fc(x)

        return output



class LSTMForecaster(nn.Module):
    """用于时间序列预测的 LSTM 模型"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=90, dropout=0.2):
        """
        初始化 LSTM 模型结构

        参数:
        - input_size: 每个时间步输入的特征维度（例如多个传感器的值）
        - hidden_size: LSTM 每层隐藏状态的维度
        - num_layers: LSTM 层数（可堆叠）
        - dropout: Dropout 比例（防止过拟合）
        """
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 构建 LSTM 模块，设置 batch_first=True 以支持 [batch, time, feature] 输入格式
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # Dropout 层用于正则化
        self.dropout = nn.Dropout(dropout)

        # 全连接层将最后时刻的隐藏状态映射为多个输出值看，短期90长期365
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播过程

        输入:
        - x: 输入张量，形状为 [batch_size, sequence_length, input_size]

        返回:
        - output: 预测值张量，形状为 [batch_size, 1]
        """
        lstm_out, _ = self.lstm(x)  # LSTM 输出 shape: [batch, seq_len, hidden_size]
        last_output = lstm_out[:, -1, :]  # 取最后一个时间步的输出（最具代表性的状态）
        dropped = self.dropout(last_output)  # 加 Dropout
        output = self.fc(dropped)  # 映射到输出值
        return output


class TransformerForecaster(nn.Module):
    """
    基于 Transformer 的时间序列预测模型 (已修改为多步预测)
    """

    def __init__(self, input_size, output_size=90, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        """
        初始化 Transformer 预测模型

        参数:
        - input_size: 每个时间步的输入特征维度
        - output_size: 预测未来时间步的数量 (例如，短期预测90个点)
        - d_model: Transformer 编码器的内部表示维度
        - nhead: 多头注意力的头数
        - num_layers: 编码器层数
        - dropout: Dropout 比例
        """
        super(TransformerForecaster, self).__init__()


        # 输入映射：将输入特征维度投影到 d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码（可学习）：添加时间位置的先验
        # 注意: 对于非常长或长度不固定的序列，固定的正弦/余弦位置编码可能更优
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))

        # 构建 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 【改动 2】: 修改输出层，将 d_model 维度的表示映射到 output_size 个预测值
        self.fc = nn.Linear(d_model, output_size)


    def forward(self, x):
        """
        前向传播

        输入:
        - x: 输入时间序列，形状为 [batch_size, seq_len, input_size]

        返回:
        - output: 预测值，形状为 [batch_size, output_size]  <--- 【改动 3】: 返回值的形状已改变
        """
        seq_len = x.size(1)

        # 1. 输入映射到 d_model 空间
        x = self.input_projection(x)  # -> [batch_size, seq_len, d_model]

        # 2. 添加位置编码
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)

        # 3. 输入 Transformer 编码器
        x = self.transformer(x)  # -> [batch_size, seq_len, d_model]

        # 4. 取最后一个时间步的输出作为整个序列的表示 (与你的 LSTM 模型逻辑一致)
        x = x[:, -1, :]  # -> [batch_size, d_model]
        x = self.dropout(x)

        # 5. 通过全连接层映射到最终的多步预测输出
        output = self.fc(x)  # -> [batch_size, output_size]

        return output


class TimeSeriesDataset(Dataset):
    """
    自定义 PyTorch 数据集类，用于处理时间序列数据。
    此版本支持多步预测 (Multi-step Forecasting)。
    """

    def __init__(self, data, sequence_length, prediction_horizon=90, target_col='global_active_power'):
        """
        参数:
        - data (pd.DataFrame): 已排序和标准化的时间序列数据。
        - sequence_length (int): 输入序列的长度 (滑动窗口大小)。
        - prediction_horizon (int): 输出序列的长度，即需要预测的未来时间步数量。
                                   短期90，长期365，即单步预测。
        - target_col (str): 需要预测的目标列名称。
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col

    def __len__(self):
        """
        计算并返回数据集中可生成的样本总数。
        现在，总长度必须减去输入序列和输出序列的长度。
        """
        # 为了生成一个完整的样本，我们需要 sequence_length 的输入和 prediction_horizon 的输出
        # 因此最后一个样本的起始点 (idx) 必须保证其后有足够的 (sequence_length + prediction_horizon) 数据点
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        """
        根据索引 `idx`，提取并返回一个 (输入序列, 目标序列) 样本对。

        返回:
        - x: 从 idx 开始的长度为 sequence_length 的输入序列 (特征)。
        - y: 紧跟在输入序列之后的、长度为 prediction_horizon 的目标序列。
        """
        # 提取输入序列 x (与之前完全相同)
        start_idx_x = idx
        end_idx_x = start_idx_x + self.sequence_length
        x = self.data.iloc[start_idx_x:end_idx_x].values

        # 提取输出/目标序列 y
        start_idx_y = end_idx_x
        end_idx_y = start_idx_y + self.prediction_horizon

        # 我们只从目标列中提取 y
        y = self.data[self.target_col].iloc[start_idx_y:end_idx_y].values

        # 转换为 PyTorch Tensors
        # x 的形状是 (sequence_length, num_features)
        # y 的形状现在是 (prediction_horizon,)
        return torch.FloatTensor(x), torch.FloatTensor(y)



class EnergyConsumptionAnalyzer:
    def __init__(self):
        """
        初始化能耗分析器类（已适配训练/测试集分离）：
        """
        # 原始数据
        self.train_raw = None
        self.test_raw = None

        # 处理后的数据（特征 + 目标）
        self.train_processed = None
        self.test_processed = None

        # 分离后的特征 (X) 和目标 (y)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # 存储模型、预测结果和标准化器
        self.models = {}
        self.predictions = {}  # 主要存储对测试集的预测
        self.scalers = {}  # 存储用于特征和目标的标准化器

        # 设置计算设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")



    def load_data(self, train_path: str, test_path: str):
        """
        加载训练集和测试集，并为每个数据集显示详细的加载信息。

        - 分别从 train_path 和 test_path 加载数据。
        - 将所有列名统一为小写格式。
        - 为每个数据集打印形状、列名、总条数和前5行数据预览。

        参数:
            train_path (str): 训练集 CSV 文件的路径。
            test_path (str): 测试集 CSV 文件的路径。
        """
        # --- 加载训练数据 ---
        print(f"📥 Loading training data from: {train_path}...")
        self.train_raw = pd.read_csv(train_path, sep=',')
        self.train_raw.columns = [col.lower() for col in self.train_raw.columns]

        print("\n✅ Training data loaded successfully:")
        print(f"   - Shape: {self.train_raw.shape}")
        print(f"   - Columns: {list(self.train_raw.columns)}")
        print(f"   - 数据总条数: {len(self.train_raw)}")
        print("   - 前5行数据预览:")
        # 使用 to_string() 保证在列数较多时也能完整显示
        print(self.train_raw.head().to_string())

        # 用分隔线让输出更清晰
        print("\n" + "=" * 60 + "\n")

        # --- 加载测试数据 ---
        print(f"📥 Loading testing data from: {test_path}...")
        self.test_raw = pd.read_csv(test_path, sep=',')
        self.test_raw.columns = [col.lower() for col in self.test_raw.columns]

        print("\n✅ Testing data loaded successfully:")
        print(f"   - Shape: {self.test_raw.shape}")
        print(f"   - Columns: {list(self.test_raw.columns)}")
        print(f"   - 数据总条数: {len(self.test_raw)}")
        print("   - 前5行数据预览:")
        print(self.test_raw.head().to_string())



    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [私有方法] 为给定的数据集创建所有时间、周期、滞后和滚动特征。
        """
        # 创建一个副本以避免对原始数据进行修改
        data = df.copy()

        # --- 时间索引创建 ---
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                if data['date'].isnull().any():
                    raise ValueError("❌ 'date' column contains values that could not be parsed.")
                data.set_index('date', inplace=True)
            else:
                raise KeyError("❌ Missing 'date' column to create datetime index.")

        # --- 所有特征工程逻辑 (与您提供的代码相同) ---
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day_of_week'] = data.index.dayofweek
        data['day_of_year'] = data.index.dayofyear
        data['quarter'] = data.index.quarter
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)

        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
        data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)

        if all(col in data.columns for col in ['sub_metering_1', 'sub_metering_2', 'sub_metering_3']):
            data['total_sub_metering'] = (
                    data['sub_metering_1'] + data['sub_metering_2'] + data['sub_metering_3']
            )

        target_col = 'global_active_power'
        if target_col in data.columns:
            lags_in_days = [1, 2, 3, 7, 14, 30]
            for lag in lags_in_days:
                data[f'{target_col}_lag_{lag}_day'] = data[target_col].shift(lag)

            windows_in_days = [3, 7, 14, 30]
            for window in windows_in_days:
                data[f'{target_col}_rolling_mean_{window}_day'] = data[target_col].rolling(window).mean()
                data[f'{target_col}_rolling_std_{window}_day'] = data[target_col].rolling(window).std()

        return data



    def preprocess_data(self, target_col: str = 'global_active_power'):
        """
        对训练集和测试集进行完整的预处理流程：
        1. 对两个数据集应用特征工程。
        2. 清理因特征工程产生的NaN值。
        3. 分离特征(X)和目标(y)。
        4. 基于训练集进行数据标准化，并应用到测试集，以防止数据泄露。
        """
        print("\n🔧 Preprocessing data for both training and testing sets...")

        # --- 1. 特征工程 ---
        # 对训练集和测试集分别调用特征创建函数
        self.train_processed = self._create_features(self.train_raw)
        self.test_processed = self._create_features(self.test_raw)
        print("   - Feature engineering applied to both sets.")

        # --- 2. 清理 NaN ---
        # 删除因滞后/滚动特征产生的NaN行
        self.train_processed.dropna(inplace=True)
        self.test_processed.dropna(inplace=True)
        print(
            f"   - NaN values dropped.  train shape: {self.train_processed.shape}, Test shape: {self.test_processed.shape}")

        # --- 3. 分离特征 (X) 和目标 (y) ---
        self.y_train = self.train_processed[[target_col]]
        self.X_train = self.train_processed.drop(columns=[target_col])
        self.y_test = self.test_processed[[target_col]]
        self.X_test = self.test_processed.drop(columns=[target_col])
        print("   - Features (X) and target (y) have been separated.")

        # --- 4. 数据标准化 (防止数据泄露的关键步骤) ---
        # 创建特征标准化器
        feature_scaler = MinMaxScaler()
        # **只用训练集特征来学习**缩放参数，然后应用到训练集
        self.X_train = pd.DataFrame(feature_scaler.fit_transform(self.X_train), columns=self.X_train.columns,
                                    index=self.X_train.index)
        # **用同一个（已经学习好的）标准化器**来转换测试集特征
        self.X_test = pd.DataFrame(feature_scaler.transform(self.X_test), columns=self.X_test.columns,
                                   index=self.X_test.index)
        self.scalers['features'] = feature_scaler  # 保存，用于未来逆向转换

        # 对目标变量也进行标准化
        target_scaler = MinMaxScaler()
        self.y_train = pd.DataFrame(target_scaler.fit_transform(self.y_train), columns=self.y_train.columns,
                                    index=self.y_train.index)
        self.y_test = pd.DataFrame(target_scaler.transform(self.y_test), columns=self.y_test.columns,
                                   index=self.y_test.index)
        self.scalers['target'] = target_scaler  # 保存，用于逆向转换预测结果

        print("   - Data scaling complete (fit on train, transform on both).")
        print("✅ Preprocessing finished. Data is ready for model training.")



    def prepare_sequences(self, data, sequence_length=90, target_col='global_active_power'):
        """
        为时间序列建模准备数据：
        - 将输入特征与目标列分开进行标准化（StandardScaler）
        - 返回标准化后的数据 DataFrame 及特征列名列表
        - 不直接切分序列，而是作为序列构造前的预处理步骤

        参数：
            data (pd.DataFrame): 已经预处理过的特征数据
            sequence_length (int): 每个样本包含的时间步数量（用历史90天数据做预测）
            target_col (str): 目标变量列名（默认为 'global_active_power'）

        返回：
            scaled_data (pd.DataFrame): 标准化后的完整数据（包含特征 + 目标）
            feature_cols (list): 所有特征列名（不含目标列）
        """

        # 🧩 Step 1: 提取特征列（排除目标列）
        feature_cols = [col for col in data.columns if col != target_col]

        # 🧪 Step 2: 初始化两个标准化器
        # - 一个用于特征
        # - 一个用于目标变量
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        # 🧼 Step 3: 对特征列和目标列分别进行标准化
        scaled_features = feature_scaler.fit_transform(data[feature_cols])
        scaled_target = target_scaler.fit_transform(data[[target_col]])

        # 🧱 Step 4: 将标准化后的特征和目标列组合成一个 DataFrame
        scaled_data = pd.DataFrame(
            np.hstack([scaled_features, scaled_target]),  # 水平堆叠
            columns=feature_cols + [target_col],  # 保持列名一致
            index=data.index  # 保留时间索引
        )

        # 💾 Step 5: 保存标准化器，用于后续模型预测时反标准化（inverse_transform）
        self.scalers['feature_scaler'] = feature_scaler
        self.scalers['target_scaler'] = target_scaler

        # ✅ 返回标准化后的数据和特征列名列表
        return scaled_data, feature_cols



    def train_pytorch_model(self, model_type='lstm', sequence_length=90, prediction_horizon=90, epochs=50, batch_size=32):
        """
        使用 PyTorch 训练用于时间序列预测的模型

        参数：
            model_type (str): 模型类型，可选 'lstm', 'transformer'，决定使用哪种预测模型。
            sequence_length (int): 输入序列长度，决定每次预测使用多少步历史数据。
            prediction_horizon (int): 预测序列长度，决定每次预测多少天的数据，短期90，长期365。
            epochs (int): 训练的最大轮数。
            batch_size (int): 批量大小。

        功能：
            - 准备训练和验证数据集
            - 初始化并训练指定的预测模型
            - 使用 MSELoss 进行回归任务优化
            - 使用学习率调度和早停机制防止过拟合
            - 在训练结束后加载验证集损失最低的模型
            - 可视化训练和验证损失变化趋势
        """

        print(f"\n🧠 Training {model_type.upper()} model...")

        # ===============================
        # 1️⃣ 数据预处理与序列生成
        # ===============================
        # 对预处理后的完整数据进行归一化并生成序列，以便输入模型进行时序预测
        scaled_data1, feature_cols1 = self.prepare_sequences(self.train_processed, sequence_length)
        scaled_data2, feature_cols2 = self.prepare_sequences(self.train_processed, sequence_length)

        # 创建用于时间序列的 Dataset 类，返回形如 (batch_x, batch_y) 的迭代器
        train_dataset = TimeSeriesDataset(scaled_data1, sequence_length=sequence_length, prediction_horizon=prediction_horizon)
        val_dataset = TimeSeriesDataset(scaled_data2, sequence_length=sequence_length, prediction_horizon=prediction_horizon)

        # ===============================
        # 2️⃣ 划分训练集和验证集
        # ===============================

        # 创建 PyTorch DataLoader，便于分批加载数据
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # ===============================
        # 3️⃣ 初始化指定类型的模型
        # ===============================
        input_size = len(feature_cols1) + 1  # 输入特征数量（特征列数 + 目标列）

        if model_type == 'lstm':
            model = LSTMForecaster(input_size=input_size,output_size=prediction_horizon).to(self.device)
        elif model_type == 'transformer':
            model = TransformerForecaster(input_size=input_size, output_size=prediction_horizon).to(self.device)
        elif model_type == 'timemixer':
            model = TimeMixer(input_size=input_size, output_size=prediction_horizon).to(self.device)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # ===============================
        # 4️⃣ 定义损失函数、优化器和学习率调度器
        # ===============================
        criterion = nn.MSELoss()  # 均方误差用于回归损失
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam 优化器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(  # 动态调整学习率
            optimizer,
            patience=5,
            factor=0.5
        )

        # ===============================
        # 5️⃣ 开始训练循环
        # ===============================
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')  # 用于记录最佳验证损失
        patience_counter = 0  # 用于早停

        for epoch in range(epochs):
            # 🔹训练阶段
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()  # 梯度清零
                outputs = model(batch_x)  # 前向传播
                loss = criterion(outputs, batch_y)  # 计算损失
                loss.backward()  # 反向传播

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()  # 更新参数
                train_loss += loss.item()  # 累加训练损失

            # 🔹验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item()

            # 🔹记录平均损失
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)  # 根据验证损失调整学习率

            # 🔹早停机制与模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{model_type}_model.pth')
            else:
                patience_counter += 1

            # 每 10 轮打印一次进度
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # 当验证集损失多次不再提升时，提前停止训练
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

        # ===============================
        # 6️⃣ 加载验证损失最低的模型参数
        # ===============================
        model.load_state_dict(torch.load(f'best_{model_type}_model.pth'))
        self.models[model_type] = model

        # ===============================
        # 7️⃣ 可视化训练与验证损失变化曲线
        # ===============================
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{model_type.upper()} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        return model



    def evaluate_models(self, prediction_horizon=90):
        """
        对所有已训练的 PyTorch 模型进行预测评估

        功能：
            - 使用训练好的模型对测试集进行预测
            - 将预测值反归一化回真实单位
            - 计算 MSE、MAE、RMSE、MAPE 四项评价指标
            - 输出对比表格及可视化柱状图
        """

        print("\n📊 Evaluating models...")

        results = {}  # 用于存储每个模型的评估结果

        # ==============================================
        # 1️⃣ 遍历需要评估的 PyTorch 模型
        # ==============================================
        for model_name in ['lstm', 'transformer', 'timemixer']:
            if model_name in self.models:
                try:
                    # ==============================================
                    # 1.1 准备测试集的滑动窗口输入
                    # ==============================================
                    scaled_data, feature_cols = self.prepare_sequences(self.test_processed, sequence_length=90)
                    test_dataset = TimeSeriesDataset(scaled_data, sequence_length=90, prediction_horizon=prediction_horizon)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                    model = self.models[model_name]
                    model.eval()  # 切换至评估模式

                    predictions = []
                    actuals = []

                    # ==============================================
                    # 1.2 模型预测并收集预测值和真实值
                    # ==============================================
                    with torch.no_grad():
                        for batch_x, batch_y in test_loader:
                            batch_x = batch_x.to(self.device)
                            outputs = model(batch_x)
                            predictions.extend(outputs.cpu().numpy().flatten())
                            actuals.extend(batch_y.numpy().flatten())

                    # ==============================================
                    # 1.3 反归一化预测值和真实值以便计算真实误差（这样误差巨大无比）
                    # ==============================================
                    predictions = np.array(predictions).reshape(-1, 1)
                    actuals = np.array(actuals).reshape(-1, 1)

                    predictions = self.scalers['target_scaler'].inverse_transform(predictions).flatten()
                    actuals = self.scalers['target_scaler'].inverse_transform(actuals).flatten()

                    # ==============================================
                    # 1.4 计算评估指标
                    # ==============================================
                    mse = mean_squared_error(actuals, predictions)
                    mae = mean_absolute_error(actuals, predictions)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

                    results[model_name] = {
                        'MSE': mse,
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    }

                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")

        # ==============================================
        # 2️⃣ 输出结果表格与可视化
        # ==============================================
        if results:
            results_df = pd.DataFrame(results).T  # 转置以便模型名为行，指标为列

            print("\n📊 Model Evaluation Results:")
            print(results_df.round(4))

            # 绘制四个指标的柱状图便于对比
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            metrics = ['MSE', 'MAE', 'RMSE', 'MAPE']

            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                results_df[metric].plot(kind='bar', ax=ax)
                ax.set_title(f'{metric} by Model')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()




    def predict_future(self, model_name='lstm', steps=90):
        """
        使用已训练好的模型进行未来 steps 步预测

        功能：
            - 使用训练好的 LSTM/Transformer 模型进行未来滚动预测
            - 将预测结果反归一化回真实单位（kW）
            - 可视化最近一周的历史值 + 未来预测值对比曲线
            - 返回预测值和对应的未来时间索引，便于进一步分析或与真实值对比
        """

        if model_name not in self.models:
            print(f"Model {model_name} not trained yet")
            return None

        print(f"\n🔮 Making {steps} step predictions using {model_name}...")

        model = self.models[model_name]
        model.eval()  # 切换到评估模式

        # ==============================================
        # 1️⃣ 从已处理数据中获取最后一个序列（用于预测起点）
        # ==============================================
        sequence_length = 90
        scaled_data, feature_cols = self.prepare_sequences(self.test_processed, sequence_length=sequence_length)
        first_sequence = scaled_data.iloc[:sequence_length].values   # shape: (sequence_length, feature_dim)

        predictions = []  # 用于保存预测值（归一化状态）
        current_sequence = torch.FloatTensor(first_sequence).unsqueeze(0).to(self.device)
        # shape: (1, sequence_length, feature_dim)

        # ==============================================
        # 2️⃣ 直接预测 steps 天的数据
        # ==============================================
        with torch.no_grad():
            pred = model(current_sequence)  # 模型一次性预测整个序列
            predictions = pred.cpu().numpy().flatten()  # 将输出转换为可用的 NumPy 数组

        # ==============================================
        # 3️⃣ 将预测值反归一化回真实单位 (kW)
        # ==============================================
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scalers['target_scaler'].inverse_transform(predictions).flatten()


        # 4️⃣ 创建对应预测时间索引
        # ==============================================
        first_date = self.test_processed.index[0]
        future_dates = pd.date_range(start=self.test_processed.index[sequence_length] + pd.Timedelta(days=1),
                                     periods=steps, freq='D')

        # ==============================================
        # 5️⃣ 可视化结果
        # ==============================================
        plt.figure(figsize=(15, 8))

        # 1️⃣ 提取从 sequence_length 开始的数据（真实数据）
        real_data = self.test_processed['global_active_power'].iloc[sequence_length:sequence_length + steps]

        # 2️⃣ 创建合并的日期索引（真实数据和预测数据）
        combined_dates = pd.date_range(start=self.test_processed.index[sequence_length], periods=steps, freq='D')

        # 3️⃣ 合并真实数据和预测数据
        combined_data = np.concatenate([real_data.values, predictions])

        # 4️⃣ 绘制真实数据和预测数据的对比
        plt.plot(combined_dates, real_data, label='Real Data', color='blue')  # 真实数据
        plt.plot(combined_dates, predictions, label='Prediction', color='red', linestyle='--')  # 预测数据

        # 5️⃣ 设置标题和标签
        plt.title(f'Real vs. Predicted Data using {model_name.upper()}')
        plt.xlabel('Date')
        plt.ylabel('Global Active Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # ==============================================
        # 6️⃣ 返回预测结果和未来时间索引便于后续对比真实值
        # ==============================================
        return predictions, future_dates


    def run_complete_analysis(self):
        """
        🔍 运行完整的电力负荷分析流程

        包含步骤：
            1. 训练深度学习预测模型（LSTM、GRU、TCN）
            2. 训练 Prophet 时间序列模型
            3. 训练异常检测模型（Autoencoder）
            4. 多种方法检测异常（Isolation Forest、统计、Autoencoder）
            5. 评估预测模型性能（MSE、MAE、RMSE、MAPE）
            6. 使用 LSTM 进行未来 48 小时滚动预测并可视化

        用途：
            - 快速执行全流程分析，适合调试、测试与最终报告生成
        """
        print("🚀 Starting Complete Energy Consumption Analysis...")

        # ==============================================
        # 1️⃣ 训练时间序列预测模型
        # ==============================================
        print("\n" + "=" * 50)
        print("TRAINING FORECASTING MODELS")
        print("=" * 50)

        self.train_pytorch_model('lstm', epochs=30)
        #self.train_pytorch_model('gru', epochs=30)


        # ==============================================
        # 2️⃣ 评估预测模型性能
        # ==============================================
        self.evaluate_models()  # 自动评估 LSTM 的预测效果（绘图 + 打印表）


        # ==============================================
        # 3️⃣ 未来 90 天滚动预测并可视化
        # ==============================================
        print("\n" + "=" * 50)
        print("MAKING FUTURE PREDICTIONS")
        print("=" * 50)

        self.predict_future('lstm', steps=90)  # 使用 LSTM 模型预测未来90 天

        print("\n✅ Complete analysis finished!")


