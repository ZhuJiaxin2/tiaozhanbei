import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_tensor(tensor, file_path, output_folder='/home/challenge/code/visualizations', vmin=None, vmax=None):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 如果没有指定 vmin 和 vmax，则自动计算
    if vmin is None or vmax is None:
        non_zero_tensor = tensor[tensor > 0]
        if len(non_zero_tensor) > 0:
            vmin = non_zero_tensor.min().item()
            vmax = non_zero_tensor.max().item()
        else:
            vmin = 0
            vmax = 1
    
    # 生成图片
    plt.figure(figsize=(10, 5))
    plt.imshow(tensor.numpy(), cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f'Tensor Visualization: {os.path.basename(file_path)}')
    
    # 保存图片到文件
    output_file = os.path.join(output_folder, f"{os.path.basename(file_path).split('.')[0]}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"可视化图片已保存到: {output_file}")

def check_tensor_file(file_path, vmin=None, vmax=None):
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return
    
    tensor = torch.load(file_path)
    print(f"加载的 Tensor 文件：{file_path}")
    print(f"Tensor 大小：{tensor.size()}")
    print(f"Tensor 数据类型：{tensor.dtype}")
    print(f"Tensor 中的最大值：{tensor.max().item()}")
    print(f"Tensor 中的最小值：{tensor.min().item()}")
    print("Tensor 数据片段：")
    print(tensor)
    
    visualize_tensor(tensor, file_path, vmin=vmin, vmax=vmax)

if __name__ == "__main__":
    # 示例使用：控制显示数据范围
    check_tensor_file('/data/challenge/yuchang/processed_mmsi/mmsi-daily-pts-10-v2-2020/2020-01-13.pt', vmin=0, vmax=24)
