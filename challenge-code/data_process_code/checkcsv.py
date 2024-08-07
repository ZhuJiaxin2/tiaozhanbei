import torch

def check_tensor_file(tensor_file):
    """
    检查并打印 Tensor 文件中的数据。

    参数：
    tensor_file (str): Tensor 文件的路径。
    """
    # 加载 Tensor 文件
    tensor_data = torch.load(tensor_file)
    
    # 打印Tensor的类型和形状
    print(f"Loaded tensor type: {type(tensor_data)}")
    
    if isinstance(tensor_data, dict):
        for key, value in tensor_data.items():
            print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape}")
            print(f"Data sample: {value}")
    elif isinstance(tensor_data, torch.Tensor):
        print(f"Tensor shape: {tensor_data.shape}")
        print(f"Data sample: {tensor_data}")
    else:
        print("The file does not contain a recognized tensor format.")

def calculate_coordinate_range(tensor_shape, resolution=0.1):
    """
    计算给定张量形状和分辨率的经纬度坐标范围。

    参数：
    tensor_shape (tuple): 张量的形状 (height, width)。
    resolution (float): 分辨率，默认为0.1度。
    
    返回：
    tuple: (min_lat, max_lat, min_lon, max_lon)
    """
    height, width = tensor_shape
    
    # 计算纬度范围
    min_lat = -90
    max_lat = min_lat + (height - 1) * resolution
    
    # 计算经度范围
    min_lon = -180
    max_lon = min_lon + (width - 1) * resolution
    
    return (min_lat, max_lat, min_lon, max_lon)

import pandas as pd

def check_latitude_range(csv_file):
    """
    检查CSV文件中`cell_ll_lat`的最小值和最大值。

    参数：
    csv_file (str): CSV文件的路径。
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    
    # 获取`cell_ll_lat`的最小值和最大值
    min_lat = data['cell_ll_lat'].min()
    max_lat = data['cell_ll_lat'].max()
    
    return min_lat, max_lat

# 示例调用
csv_file = '/data/challenge/yuchang/mmsi/mmsi-daily-csvs-10-v2-2019/2019-01-11.csv'
min_lat, max_lat = check_latitude_range(csv_file)
print(f"Minimum cell_ll_lat: {min_lat}")
print(f"Maximum cell_ll_lat: {max_lat}")


""" # 示例调用
tensor_shape = (1801, 3601)
coordinate_range = calculate_coordinate_range(tensor_shape)
print(f"Latitude range: {coordinate_range[0]} to {coordinate_range[1]}")
print(f"Longitude range: {coordinate_range[2]} to {coordinate_range[3]}") """

""" 
# 示例调用
tensor_file = '/data/challenge/yuchang/processed_mmsi/mmsi-daily-pts-10-v2-2019/2019-01-01.pt'

# 检查 Tensor 文件内容
check_tensor_file(tensor_file) """
