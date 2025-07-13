import pandas as pd
import os
import numpy as np

import pandas as pd
import os
import numpy as np


class DataPreprocessor:
    """
    一个用于预处理家庭功耗时间序列数据的完整类。
    新增了：
    1. 对数据精度的统一处理。
    2. 将最终输出的索引列名改为 'date'。
    """

    def __init__(self, input_filepath: str, output_filepath: str, final_precision: int = 4):
        """
        初始化预处理器。

        参数:
        input_filepath (str): 输入的 .csv 文件路径。
        output_filepath (str): 处理后要保存的 .csv 文件路径。
        final_precision (int): 最终输出数据时保留的小数位数。
        """
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"输入文件未找到： {input_filepath}")

        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.final_precision = final_precision
        self.data = None
        print(f"预处理器已初始化。输入文件：'{self.input_filepath}', 输出文件：'{self.output_filepath}'")
        print(f"最终数据精度将统一为 {self.final_precision} 位小数。")

    def load_and_prepare(self):
        """加载数据，处理缺失值符号，转换列名，设置索引。"""
        print("\n--- 步骤 1: 加载数据并设置索引 ---")
        try:
            self.data = pd.read_csv(
                self.input_filepath,
                parse_dates=['DateTime'],
                index_col='DateTime',
                na_values=['?', '']
            )
            print("✅ 数据加载成功！")
            self.data.columns = [col.lower() for col in self.data.columns]
            print("✅ 所有列名已转换为小写。")
            print("\n加载后各列的缺失值数量:")
            print(self.data.isnull().sum())
        except Exception as e:
            print(f"❌ 加载数据时发生错误: {e}")
            self.data = None
        return self

    def _handle_missing_values(self, strategy='ffill'):
        """处理数据中的缺失值 (NaN)。"""
        if self.data is None: return
        print(f"\n--- 步骤 2a: 处理缺失值 (策略: {strategy}) ---")
        if self.data.isnull().sum().sum() > 0:
            self.data.fillna(method=strategy, inplace=True)
            print(f"✅ 已使用'{strategy}'策略处理所有缺失值。")
        else:
            print("✅ 数据中没有发现缺失值，无需处理。")

    def _create_features(self):
        """根据公式创建新的特征列。"""
        if self.data is None: return
        print("\n--- 步骤 2b: 创建新特征 'sub_metering_remainder' ---")
        try:
            self.data['sub_metering_remainder'] = \
                (self.data['global_active_power'] * 1000 / 60) - \
                (self.data['sub_metering_1'] + self.data['sub_metering_2'] + self.data['sub_metering_3'])

            cols = self.data.columns.tolist()
            insert_pos = cols.index('sub_metering_3') + 1
            cols.insert(insert_pos, cols.pop(cols.index('sub_metering_remainder')))
            self.data = self.data.loc[:, cols]
            print("✅ 新特征 'sub_metering_remainder' 创建成功。")
        except Exception as e:
            print(f"❌ 创建特征时发生错误: {e}")

    def _resample_data(self):
        """将数据按天降采样，并重命名索引。"""
        if self.data is None: return
        print("\n--- 步骤 2c: 按天降采样数据 ---")
        try:
            aggregation_rules = {
                'global_active_power': 'sum', 'global_reactive_power': 'sum',
                'voltage': 'mean', 'global_intensity': 'mean',
                'sub_metering_1': 'sum', 'sub_metering_2': 'sum',
                'sub_metering_3': 'sum', 'sub_metering_remainder': 'sum',
                'rr': 'first', 'nbjrr1': 'first', 'nbjrr5': 'first',
                'nbjrr10': 'first', 'nbjbrou': 'first'
            }
            self.data = self.data.resample('D').agg(aggregation_rules)
            print("✅ 数据已成功按天聚合。")

            # 【核心改动】重命名索引列
            self.data.index.name = 'date'
            print("✅ 索引列名已从 'DateTime' 更新为 'date'。")

        except Exception as e:
            print(f"❌ 降采样数据时发生错误: {e}")

    def _finalize_precision(self):
        """统一所有数值列的精度。"""
        if self.data is None: return
        print(f"\n--- 步骤 2d: 统一数据精度 ---")
        try:
            self.data = self.data.round(self.final_precision)
            print(f"✅ 所有数值已四舍五入，保留 {self.final_precision} 位小数。")
        except Exception as e:
            print(f"❌ 统一精度时发生错误: {e}")

    def process_data(self):
        """执行所有数据处理步骤的流水线。"""
        if self.data is None:
            print("数据尚未加载，请先调用 load_and_prepare()。")
            return self

        print("\n--- 步骤 2: 开始执行数据处理流程 ---")
        self._handle_missing_values(strategy='ffill')
        self._create_features()
        self._resample_data()
        self._finalize_precision()

        print("\n✅ 所有数据处理步骤完成。")
        return self

    def save_to_csv(self):
        """将处理后的数据保存到指定的输出 .csv 文件。"""
        if self.data is None:
            print("没有数据可以保存。")
            return
        print(f"\n--- 步骤 3: 保存处理后的数据 ---")
        try:
            # to_csv 会自动使用更新后的索引名 'date'
            self.data.to_csv(self.output_filepath)
            print(f"✅ 数据已成功保存到: {self.output_filepath}")
            print("\n处理后数据预览 (前5行):")
            print(self.data.head())
        except Exception as e:
            print(f"❌ 保存文件时发生错误: {e}")



# --- 如何使用这个类 ---
if __name__ == '__main__':

    input_filename = '../data/origin_data/train.csv'


    output_filename = '../data/preprocessed_data/preprocessed_train.csv'

    print("--- 开始预处理流程 ---")

    # **核心流程**
    # 1. 创建类的实例, 可以指定 final_precision 参数
    preprocessor = DataPreprocessor(
        input_filepath=input_filename,
        output_filepath=output_filename,
        final_precision=4  # 指定最终保留4位小数
    )

    # 2. 执行处理流程
    preprocessor.load_and_prepare() \
        .process_data() \
        .save_to_csv()

    print("\n--- 预处理流程执行完毕 ---")
