# import torch

# file_path = '/data/challenge_data/combined_data/processed_data_20190101.pt'
# data = torch.load(file_path)
# print(data.shape)
import matplotlib.pyplot as plt
import numpy as np

# 定义x和y轴的标记
x_labels = ["0", "200", "400", "600", "800", "1000", "1200"]
y_labels = ["0", "200", "400", "600"]

# 创建一个2D矩阵
values = np.array([[0.2, 0.3, 1.5, 0.5, 0.6],
                   [2.5, 0.7, 0.7, 1.8, 0.9],
                   [0.5, 1.4, 2.5, 0.5, 1.5],
                   [0.4, 1.9, 2.0, 0.5, 0.5],
                   [1.0, 2.4, 0.1, 0.0, 1.5]])

# 绘制热力图
fig, axe = plt.subplots(figsize=(8, 5))
axe.set_xticks(np.arange(len(x_labels)))
axe.set_yticks(np.arange(len(y_labels)))
axe.set_xticklabels(x_labels)
axe.set_yticklabels(y_labels)
im = axe.imshow(values)

# 显示图像
plt.show()
