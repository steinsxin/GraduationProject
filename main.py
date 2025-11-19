import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py


class MatDataVisualizer:
    """
    MAT 数据可视化工具类
    - 智能读取 .mat 文件
    - 打印数据集信息
    - 绘制指定索引的样本
    """

    def __init__(self, mat_file, save_dir='plots'):
        """
        初始化

        :param mat_file: str, .mat 文件路径
        :param save_dir: str, 图表保存目录
        """
        self.mat_file = mat_file
        self.save_dir = save_dir
        self.data = None

        # Matplotlib 中文显示设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def load_mat_smart(self):
        """
        智能读取任何版本的 .mat 文件
        支持 v7.2 及更早版本
        """
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
                f"数据集 '{key}' 不存在，请先调用 load_mat_smart()"
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
        plt.xlabel('特征索引', fontsize=12)
        plt.ylabel('特征值', fontsize=12)
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
                f"数据集 '{key}' 不存在，请先调用 load_mat_smart()"
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
                title = (f"{key} 第 {idx} 个样本 - {sample.size} 特征")

                self._create_sample_plot(sample, title)
                save_path = self._save_plot(f'{key}_sample_{idx}.png')
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
        """
        创建组合图表

        :param arr: array, 数据数组
        :param key: str, 数据集名称
        :param sample_indices: list, 样本索引列表
        """
        plt.figure(figsize=(20, 6 * len(sample_indices)))

        for i, idx in enumerate(sample_indices, 1):
            sample = arr[idx]
            plt.subplot(len(sample_indices), 1, i)
            plt.plot(sample, linewidth=1)
            plt.title(
                f"{key} 第 {idx} 个样本 - {sample.size} 特征",
                fontsize=12,
                fontweight='bold'
            )
            plt.xlabel('特征索引', fontsize=10)
            plt.ylabel('特征值', fontsize=10)
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
    """主函数示例"""
    mat_file_path = 'CPSC2025_Dataset/CPSC2025_data/train/traindata.mat'

    visualizer = MatDataVisualizer(mat_file_path)
    visualizer.load_mat_smart()
    visualizer.print_dataset_info('traindata')

    saved_paths = visualizer.plot_sample(
        'traindata',
        [0, 500],
        save_individual=True,
        save_combined=True
    )
    print("保存的路径:", saved_paths)

if __name__ == '__main__':
    main()