import os
import pandas as pd
import numpy as np
import torch

def read_and_aggregate_data(file_path):
    # 读取单个 CSV 文件
    data = pd.read_csv(file_path)
    
    # 使用 groupby 进行坐标点和 fishing_hours 的叠加
    aggregated_data = data.groupby(['cell_ll_lat', 'cell_ll_lon'], as_index=False)['fishing_hours'].sum()
    
    return aggregated_data

def convert_to_tensor(aggregated_data, resolution=0.1):
    # 确定坐标范围（可以根据具体数据调整）
    min_lat, max_lat = -90, 90
    min_lon, max_lon = -180, 180
    
    # 创建一个坐标映射到 fishing_hours 的字典
    coord_to_fishing_hours = {(row['cell_ll_lat'], row['cell_ll_lon']): row['fishing_hours'] for _, row in aggregated_data.iterrows()}
    
    # 计算网格大小
    lat_bins = int((max_lat - min_lat) / resolution) + 1
    lon_bins = int((max_lon - min_lon) / resolution) + 1
    
    # 初始化一个全零的矩阵
    tensor = np.zeros((lat_bins, lon_bins))
    
    # 将 fishing_hours 值填入矩阵
    for (lat, lon), fishing_hours in coord_to_fishing_hours.items():
        lat_idx = int((lat - min_lat) / resolution)
        lon_idx = int((lon - min_lon) / resolution)
        tensor[lat_idx, lon_idx] += fishing_hours  # 注意是叠加
    
    # 转换为 PyTorch Tensor
    tensor = torch.tensor(tensor, dtype=torch.float32)
    
    return tensor

def save_tensor(tensor, file_path):
    torch.save(tensor, file_path)

def process_data_for_day(file_path, save_folder, resolution=0.1):
    # 读取并聚合数据
    aggregated_data = read_and_aggregate_data(file_path)
    
    # 转换为 Tensor
    tensor = convert_to_tensor(aggregated_data, resolution)
    
    # 提取日期信息
    file_name = os.path.basename(file_path)
    date_str = file_name.split('.')[0]
    
    # 保存 Tensor 的路径
    tensor_save_path = os.path.join(save_folder, f"{date_str}.pt")
    
    # 保存 Tensor
    save_tensor(tensor, tensor_save_path)

def process_year_data(year, resolution=0.1):
    # 输入文件夹路径
    folder_path = f'/data/challenge_data/yuchang/mmsi/mmsi-daily-csvs-10-v2-{year}/'
    
    # 输出文件夹路径
    save_folder = f'/data/challenge_data/yuchang/processed_mmsi/mmsi-daily-pts-10-v2-{year}/'
    
    # 检查输入文件夹路径是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹路径不存在：{folder_path}")
        return
    
    # 检查输出文件夹路径，如果不存在则创建
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 获取所有 CSV 文件路径
    file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.csv')]
    
    # 逐个处理每个 CSV 文件
    for file_path in file_paths:
        process_data_for_day(file_path, save_folder, resolution)

def main():
    # 处理 2019 年的数据，空间分辨率为 0.1 度
    process_year_data(2019, resolution=0.1)
    
    # 处理 2020 年的数据，空间分辨率为 0.1 度
    process_year_data(2020, resolution=0.1)

    print("finish!")

if __name__ == "__main__":
    main()
