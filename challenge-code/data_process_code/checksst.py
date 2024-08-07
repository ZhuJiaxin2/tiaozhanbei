import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def check_and_visualize_tensor(tensor_file, output_dir):
    """
    检查并可视化保存的 Tensor 文件。

    参数：
    tensor_file (str): Tensor 文件的路径。
    output_dir (str): 输出图像的目录。
    """
    # 加载 Tensor 文件
    tensor_data = torch.load(tensor_file)
    
    # 提取白天和夜晚的 SST 数据
    sst_day = tensor_data[:, :, 0]
    sst_night = tensor_data[:, :, 1]
    
    # 打印数据形状
    print(f"Shape of SST day tensor: {sst_day.shape}")
    print(f"Shape of SST night tensor: {sst_night.shape}")
    
    # 将填充值 (-32768) 替换为 NaN
    fill_value = -32768.0
    sst_day = sst_day.numpy()
    sst_night = sst_night.numpy()
    sst_day = np.where(sst_day == fill_value, np.nan, sst_day)
    sst_night = np.where(sst_night == fill_value, np.nan, sst_night)
    
    # 打印部分数据进行验证
    print(f"SST day tensor sample: {sst_day[:5, :5]}")
    print(f"SST night tensor sample: {sst_night[:5, :5]}")
    
    # 检查 SST 数据的最小值和最大值
    sst_day_min = np.nanmin(sst_day)
    sst_day_max = np.nanmax(sst_day)
    sst_night_min = np.nanmin(sst_night)
    sst_night_max = np.nanmax(sst_night)
    print(f"SST day tensor min value: {sst_day_min}, max value: {sst_day_max}")
    print(f"SST night tensor min value: {sst_night_min}, max value: {sst_night_max}")
    
    # 可视化 SST 数据
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(sst_day, cmap='coolwarm')
    plt.colorbar(label='SST (Day)')
    plt.title('SST (Day)')
    
    plt.subplot(1, 2, 2)
    plt.imshow(sst_night, cmap='coolwarm')
    plt.colorbar(label='SST (Night)')
    plt.title('SST (Night)')
    
    # 保存图像
    output_file = os.path.join(output_dir, os.path.basename(tensor_file).replace('.pt', '.png'))
    plt.savefig(output_file)
    plt.show()
    print(f"SST visualization saved to {output_file}")

# 示例调用
tensor_file = '/data/challenge/processed_sst/2020/sst_20201002.pt'
output_dir = '/home/challenge/code/visualizations'
check_and_visualize_tensor(tensor_file, output_dir)
