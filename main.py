import scipy.io as sio
import h5py
import numpy as np

def load_mat_smart(filename):
    """智能读取任何版本的 .mat 文件"""
    try:
        # 先尝试用 scipy 读取（v7.2及更早版本）
        data = sio.loadmat(filename)
        # 过滤掉 MATLAB 元数据
        return {k: v for k, v in data.items() if not k.startswith('__')}
    except NotImplementedError:
        with h5py.File(filename, 'r') as f:
            data = {}
            for key in f.keys():
                # 读取并转换数据
                dataset = f[key]
                if dataset.shape == ():  # 标量
                    data[key] = dataset[()]
                else:  # 数组
                    data[key] = np.array(dataset).T  # 转置修正维度
            return data

# 使用示例
mat_data = load_mat_smart('CPSC2025_Dataset/CPSC2025_data/traind/traindata.mat')
print("变量:", mat_data.keys())

traindata = mat_data['traindata']

# 打印完整信息
print("="*50)
print("数据集详细信息")
print("="*50)

# 1. 基本属性
print(f"数据类型: {type(traindata)}")
print(f"数据dtype: {traindata.dtype}")

# 2. 形状信息（最关键）
print(f"\n数据形状: {traindata.shape}")

# 解读形状（根据维度数量）
if traindata.ndim == 1:
    print(f"  → 这是一个1维数组，包含 {traindata.shape[0]} 个样本")
elif traindata.ndim == 2:
    print(f"  → 样本数量: {traindata.shape[0]}")
    print(f"  → 每个样本的特征数: {traindata.shape[1]}")
elif traindata.ndim == 3:
    print(f"  → 样本数量: {traindata.shape[0]}")
    print(f"  → 每个样本的时间步/序列长度: {traindata.shape[1]}")
    print(f"  → 每个时间步的特征数: {traindata.shape[2]}")
else:
    print(f"  → 高维数据: {traindata.ndim} 维")

# 3. 数据大小
print(f"\n数据总大小: {traindata.size} 个元素")
print(f"内存占用: ~{traindata.nbytes / 1024**2:.2f} MB")

# 4. 数值统计
print(f"\n数值统计:")
print(f"  - 最小值: {traindata.min():.4f}")
print(f"  - 最大值: {traindata.max():.4f}")
print(f"  - 均值: {traindata.mean():.4f}")
print(f"  - 标准差: {traindata.std():.4f}")

# 5. 内容预览
print(f"\n前3个样本的预览:")
if traindata.ndim == 1:
    print(traindata[:3])
elif traindata.ndim >= 2:
    print(traindata[:3])  # 显示前3个样本