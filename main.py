import os
import sys
# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py

from MIT_BIH.data_processing.Dealdata import ECG_Datadeal


class MatDataVisualizer:
    """
    MAT 数据可视化工具类
    - 智能读取 .mat 文件
    - 打印数据集信息
    - 绘制指定索引的样本
    """

    def __init__(self, mat_file=None, save_dir='plots'):
        """
        初始化

        :param mat_file: str, .mat 文件路径
        :param save_dir: str, 图表保存目录
        """
        self.mat_file = mat_file
        self.save_dir = save_dir
        self.data = None

    def load_mat_smart(self):
        """
        智能读取任何版本的 .mat 文件
        支持 v7.2 及更早版本
        """
        if not self.mat_file:
            raise ValueError("mat_file path must be provided to load .mat file.")
        try:
            mat_data = sio.loadmat(self.mat_file)
            # 过滤 MATLAB 元数据
            self.data = {
                k: v for k, v in mat_data.items()
                if not k.startswith('__')
            }
        except NotImplementedError:
            # 处理 v7.3 及以上（HDF5 格式）
            with h5py.File(self.mat_file, 'r') as f:
                self.data = {}
                for key in f.keys():
                    dataset = f[key]
                    if dataset.shape == ():  # 标量
                        self.data[key] = dataset[()]
                    else:
                        self.data[key] = np.array(dataset).T  # 转置修正维度

        return self.data

    def print_dataset_info(self, key):
        """
        打印指定数据集详细信息

        :param key: str, 数据集名称
        """
        if self.data is None or key not in self.data:
            raise ValueError(
                f"数据集 '{key}' 不存在，请先加载数据"
            )

        arr = self.data[key]
        print("=" * 50)
        print(f"数据集 '{key}' 详细信息")
        print("=" * 50)
        print(f"数据类型: {type(arr)}")
        print(f"数据 dtype: {arr.dtype}")
        print(f"\n数据形状: {arr.shape}")

        if arr.ndim == 1:
            print(f"  → 1维数组，包含 {arr.shape[0]} 个样本")
        elif arr.ndim == 2:
            print(f"  → 样本数量: {arr.shape[0]}")
            print(f"  → 每个样本的特征数: {arr.shape[1]}")
        elif arr.ndim == 3:
            print(f"  → 样本数量: {arr.shape[0]}")
            print(f"  → 每个样本的时间步/序列长度: {arr.shape[1]}")
            print(f"  → 每个时间步的特征数: {arr.shape[2]}")
        else:
            print(f"  → 高维数据: {arr.ndim} 维")

        print(f"\n数据总大小: {arr.size} 个元素")
        print(f"内存占用: ~{arr.nbytes / 1024 ** 2:.2f} MB")
        print("\n数值统计:")
        print(f"  - 最小值: {arr.min():.4f}")
        print(f"  - 最大值: {arr.max():.4f}")
        print(f"  - 均值: {arr.mean():.4f}")
        print(f"  - 标准差: {arr.std():.4f}")

        print("\n前3个样本预览:")
        if arr.ndim == 1:
            print(arr[:3])
        elif arr.ndim >= 2:
            print(arr[:3])

    def _save_plot(self, filename, dpi=300, facecolor='white'):
        """
        保存当前 matplotlib 图像

        :param filename: str, 文件名
        :param dpi: int, 图像分辨率
        :param facecolor: str, 背景颜色
        :return: str, 完整的保存路径
        """
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, filename)

        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor=facecolor
        )

        abs_path = os.path.abspath(save_path)
        print(f"✅ 图表已保存至: {abs_path}")
        return abs_path

    def _create_sample_plot(self, sample, title, figsize=(20, 6)):
        """
        创建样本绘图

        :param sample: array, 样本数据
        :param title: str, 图表标题
        :param figsize: tuple, 图表尺寸
        :return: matplotlib figure 对象
        """
        plt.figure(figsize=figsize)
        plt.plot(sample, linewidth=1)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Feature Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def plot_sample(self, key, sample_indices, save_individual=True,
                    save_combined=True):
        """
        绘制指定数据集的样本

        :param key: str, 数据集名称
        :param sample_indices: int or list, 样本索引
        :param save_individual: bool, 是否保存单个样本图
        :param save_combined: bool, 是否保存组合图
        :return: dict, 保存的文件路径
        """
        if self.data is None or key not in self.data:
            raise ValueError(
                f"数据集 '{key}' 不存在，请先加载数据"
            )

        arr = self.data[key]

        # 统一处理为列表
        if isinstance(sample_indices, int):
            sample_indices = [sample_indices]

        # 验证索引有效性
        for idx in sample_indices:
            if idx >= arr.shape[0]:
                raise ValueError(
                    f"索引 {idx} 超出范围，数据集只有 {arr.shape[0]} 个样本"
                )

        saved_paths = {}

        # 绘制单个样本图
        if save_individual:
            individual_paths = []
            for idx in sample_indices:
                sample = arr[idx]
                title = f"{key} Sample {idx}  ({sample.size} features)"

                self._create_sample_plot(sample, title)
                # Ensure the filename is different for processed data
                filename = f'processed_sample_{idx}.png' if 'processed' in key else f'{key}_sample_{idx}.png'
                save_path = self._save_plot(filename)
                plt.show()

                individual_paths.append(save_path)

            saved_paths['individual'] = individual_paths

        # 绘制组合图
        if save_combined and len(sample_indices) > 1:
            self._create_combined_plot(arr, key, sample_indices)
            save_path = self._save_plot(f'{key}_samples_combined.png')
            plt.show()

            saved_paths['combined'] = save_path

        return saved_paths

    def _create_combined_plot(self, arr, key, sample_indices):
        plt.figure(figsize=(20, 6 * len(sample_indices)))
        for i, idx in enumerate(sample_indices, 1):
            sample = arr[idx]
            plt.subplot(len(sample_indices), 1, i)
            plt.plot(sample, linewidth=1)
            plt.title(f"{key} Sample {idx}  ({sample.size} features)",
                    fontsize=12, fontweight='bold')
            plt.xlabel('Feature Index', fontsize=10)
            plt.ylabel('Feature Value', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()


    def plot_first_sample(self, key):
        """
        绘制第一个样本（兼容旧版本）

        :param key: str, 数据集名称
        :return: str, 保存路径
        """
        paths = self.plot_sample(
            key, 0, save_individual=True, save_combined=False
        )
        return paths['individual'][0]

    def plot_first_and_501_sample(self, key):
        """
        绘制第一个和第501个样本

        :param key: str, 数据集名称
        :return: dict, 保存的文件路径
        """
        return self.plot_sample(key, [0, 500], save_individual=True,
                                save_combined=True)


def main():
    # 1. 定义原始数据路径和样本索引
    mat_file_path = 'CPSC2025_Dataset/CPSC2025_data/train/traindata.mat'
    sample_indices_to_plot = list(range(0, 10)) + list(range(500, 510))

    # 2. 可视化并保存原始数据
    print(">>> (1/2) 开始处理和可视化原始数据...")
    visualizer_orig = MatDataVisualizer(mat_file_path, save_dir='plots/original_data')
    original_data = visualizer_orig.load_mat_smart()
    
    print(f"\n>>> 正在绘制并保存 {len(sample_indices_to_plot)} 个原始样本...")
    for i in sample_indices_to_plot:
        visualizer_orig.plot_sample(
            'traindata',
            [i],
            save_individual=True,
            save_combined=False
        )
        plt.close('all') # 清理内存
    print('>>> 原始样本图片已保存至', os.path.abspath(visualizer_orig.save_dir))

    # 3. 处理数据
    print("\n>>> (2/2) 开始处理和可视化处理后的数据...")
    print(">>> 正在使用 ECG_Datadeal 处理数据...")
    processed_npy_path = ECG_Datadeal(mat_file_path)
    
    # 4. 加载并可视化处理后的数据
    processed_data = np.load(processed_npy_path)
    visualizer_proc = MatDataVisualizer(save_dir='plots/processed_data')
    processed_data_key = "processed_traindata"
    visualizer_proc.data = {processed_data_key: processed_data}

    print(f"\n>>> 正在绘制并保存 {len(sample_indices_to_plot)} 个处理后的样本...")
    for i in sample_indices_to_plot:
        visualizer_proc.plot_sample(
            processed_data_key,
            [i],
            save_individual=True,
            save_combined=False
        )
        plt.close('all') # 清理内存
    print('>>> 处理后的样本图片已保存至', os.path.abspath(visualizer_proc.save_dir))

    print("\n>>> 全部任务完成。")


if __name__ == '__main__':
    main()