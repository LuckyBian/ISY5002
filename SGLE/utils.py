import numpy as np
from PIL import Image
import torch

def image_from_path(image_path):
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()

    return data_lowlight

def image_from_input(input_data):
    if isinstance(input_data, str):  # 如果输入是文件路径
        data_lowlight = Image.open(input_data)
    else:  # 如果输入是numpy数组
        data_lowlight = Image.fromarray(input_data)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).unsqueeze(0) # 将channels放在前面并添加batch维度
    
    return data_lowlight



def scale_image(data_lowlight, scale_factor, device):
    h = ((data_lowlight.shape[2]) // scale_factor) * scale_factor
    w = ((data_lowlight.shape[3]) // scale_factor) * scale_factor
    data_lowlight = data_lowlight[:, :, 0:h, 0:w]
    data_lowlight = data_lowlight.to(device)
    return data_lowlight



def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
