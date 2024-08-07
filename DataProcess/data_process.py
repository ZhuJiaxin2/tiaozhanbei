import os
import datetime
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

def extract_data(start_date, end_date):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    pred_date == start_date + datetime.timedelta(days=1)

    data_dir = '../challenge_cup_backend/static/combined_data'
    save_dir = f'../challenge_cup_backend/static/extracted_{pred_date}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in os.listdir(data_dir):
        file_date = datetime.datetime.strptime(file.split('.')[0], '%Y-%m-%d')
        if file_date >= start_date and file_date <= end_date:
            shutil.copyfile(os.path.join(data_dir, file), os.path.join(save_dir, file))
        if file_date == start_date:
            shutil.copyfile(os.path.join(data_dir, file), os.path.join(save_dir, file))
        if file_date == end_date:
            shutil.copyfile(os.path.join(data_dir, file), os.path.join(save_dir, file))
    
    return pred_date


def load_data(date_str):
    # 将字符串日期转换为 datetime 对象
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 将日期格式化为年和年-月-日
    year = date.strftime('%Y')
    date_str = date.strftime('%Y-%m-%d')
    date_str_compact = date.strftime('%Y%m%d')  # 用于那些没有分隔符的日期
    # 构建文件名
    sst_file = f'../challenge_cup_backend/static/upload/sst/{year}/{date_str}.pt'
    ssha_file = f'../challenge_cup_backend/static/upload/ssha/{year}/dt_global_twosat_phy_l4_{date_str_compact}_vDT2021.pt'
    # ssha_file = f'../challenge_cup_backend/static/upload/ssha/{year}/{date_str}.pt'
    chla_file = f'../challenge_cup_backend/static/upload/chla/{year}/cmems_mod_glo_bgc_0.25deg_{date_str_compact}_chl_180.00W-179.75E_80.00S-90.00N_0.51-53.85m.pt'
    # chla_file = f'../challenge_cup_backend/static/upload/chla/{year}/{date_str}.pt'
    # 加载数据
    sst_data = torch.load(sst_file)
    ssha_data = torch.load(ssha_file)
    chla_data = torch.load(chla_file)['chl']  # 假设chl是我们需要的键

    return sst_data, ssha_data, chla_data


def data_combine(start_date, end_date):
    save_folder = '../challenge_cup_backend/static/combined_data'

    # 转换字符串日期为datetime对象
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    # 计算总天数
    total_days = (end_date - start_date).days + 1
    # 创建一个tqdm对象
    pbar = tqdm(total=total_days, desc='Processing data', ncols=80)
    # 遍历日期范围
    current_date = start_date
    # mask = torch.load('/data/challenge/mask.pt')
    while current_date <= end_date:
        # 加载当前日期的数据
        sst_data, ssha_data, chla_data = load_data(current_date)
        # 在这里对数据进行处理
        # # SST
        tensor1=sst_data
        tensor1=tensor1.unsqueeze(0).unsqueeze(0)
        tensor1=torch.nan_to_num(tensor1,nan=0.0)
        tensor1=F.adaptive_max_pool2d(tensor1,(720,1440))
        # tensor1=tensor1*mask
        # # SSHA
        tensor2=ssha_data
        tensor2=tensor2.unsqueeze(0).unsqueeze(0)
        tensor2=torch.nan_to_num(tensor2,nan=0.0)
        # tensor2=tensor2*mask
        # # CHLA
        # 假设 chl_tensor 是原始数据，形状为 [19, 681, 1440]
        tensor_chla=chla_data
        original_shape = tensor_chla.shape
        desired_shape = (original_shape[0], 720, original_shape[2])
        fill_start = 40  # 从索引40开始填充原始数据
        # 创建一个新的张量，形状为 [19, 720, 1440]，使用np.nan初始化
        new_chl_tensor = np.full(desired_shape, np.nan)
        # 将原始数据转换为numpy数组（如果原始数据是Tensor）
        if isinstance(tensor_chla, torch.Tensor):
            chl_tensor = tensor_chla.numpy()
        # 将原始数据复制到新数组的指定位置
        new_chl_tensor[:, fill_start-1:fill_start -1  + original_shape[1], :] = chl_tensor
        # 将结果转换回Tensor
        new_chl_tensor = torch.tensor(new_chl_tensor, dtype=torch.float)
        # 导入CHLA深度1
        tensor3=new_chl_tensor[0]
        tensor3=tensor3.unsqueeze(0).unsqueeze(0)
        tensor3=torch.nan_to_num(tensor3,nan=0.0)
        # tensor3=tensor3*mask
        # ###### 导入不同CHLA深度的代码

        # 假设处理后的数据是processed_data，且已经是形状为[1,3,720,1440]的张量
        processed_data=torch.cat((tensor1,tensor2,tensor3),dim=1)
        # 构建保存路径
        date_str = current_date.strftime('%Y%m%d')
        save_path = f'/data/challenge_data/{save_folder}/{date_str}.pt'
        # 保存处理后的数据到文件
        torch.save(processed_data, save_path)
        # 更新进度条
        pbar.update(1)
        # 移动到下一天
        current_date += timedelta(days=1)
    # 关闭进度条
    pbar.close()
