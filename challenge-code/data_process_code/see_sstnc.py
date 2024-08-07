import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os

def check_and_visualize_original_nc_file(nc_file, output_dir, label):
    """
    检查原始NetCDF文件中的数据，并进行可视化。

    参数：
    nc_file (str): NetCDF文件的路径。
    output_dir (str): 输出图像的目录。
    label (str): 白天或夜晚的标签（'day'或'night'）。
    """
    try:
        dataset = Dataset(nc_file, 'r')

        # 读取变量
        sst = dataset.variables['sea_surface_temperature'][:]

        # 检查数据的统计信息
        sst_min = np.nanmin(sst)
        sst_max = np.nanmax(sst)
        sst_nan_count = np.isnan(sst).sum()
        print(f"SST ({label}) min value: {sst_min}, max value: {sst_max}, NaN count: {sst_nan_count}")

        # 可视化数据
        plt.figure(figsize=(6, 6))
        plt.imshow(sst[0, :, :], cmap='coolwarm')
        plt.colorbar(label='SST')
        plt.title(f'SST ({label})')

        # 保存图像
        output_file = os.path.join(output_dir, os.path.basename(nc_file).replace('.nc', f'_{label}_original.png'))
        plt.savefig(output_file)
        plt.show()
        print(f"SST ({label}) visualization saved to {output_file}")

        dataset.close()
    except FileNotFoundError:
        print(f"File not found: {nc_file}")

# 示例调用
output_dir = '/home/challenge/code/visualizations'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

nc_file_day = '/data/challenge/sst/2020/20201002141610-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA19_G_2020276_day-v02.0-fv01.0.nc'
nc_file_night = '/data/challenge/sst/2020/20201002170930-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA19_G_2020276_night-v02.0-fv01.0.nc'

check_and_visualize_original_nc_file(nc_file_day, output_dir, 'day')
check_and_visualize_original_nc_file(nc_file_night, output_dir, 'night')
