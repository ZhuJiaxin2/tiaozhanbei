import torch
import torch.nn.functional as F

nc_file = 'dt_global_twosat_phy_l4_20190101_vDT2021.nc'
nc = Dataset(nc_file, 'r')
print(nc.variables.keys())


input_tensor = 

input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
output_tensor = F.adaptive_avg_pool2d(input_tensor, (720, 1440))  # 使用自适应平均池化调整形状
output_tensor = output_tensor.squeeze(0).squeeze(0)  # 移除批次和通道维度

print(output_tensor.shape)  # torch.Size([720, 1440])
