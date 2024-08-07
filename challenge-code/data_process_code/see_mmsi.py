import torch

# 加载 .pt 文件中的数据
data = torch.load('/data/challenge_data/yuchang/processed_mmsi/mmsi-daily-pts-10-v2-2019/2019-01-01.pt')

# 查看数据类型和结构
def print_structure(data, indent=0):
    indent_str = '  ' * indent
    if isinstance(data, dict):
        print(f"{indent_str}Dictionary with {len(data)} keys:")
        for key, value in data.items():
            print(f"{indent_str}  Key: '{key}'")
            print_structure(value, indent + 1)
    elif isinstance(data, list):
        print(f"{indent_str}List with {len(data)} elements:")
        for i, value in enumerate(data):
            print(f"{indent_str}  Index {i}:")
            print_structure(value, indent + 1)
    elif isinstance(data, torch.Tensor):
        print(f"{indent_str}Tensor of shape {data.shape} and dtype {data.dtype}")
    else:
        print(f"{indent_str}{type(data)}: {data}")

# 打印 .pt 文件的结构信息
print_structure(data)
