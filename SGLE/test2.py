import os
import glob
import time
import torch
import torchvision
from .modeling import model
from .option import *
from .utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['CUDA_VISIBLE_DEVICES']='1' # For GPU only
device = get_device()

class Tester(): 
    def __init__(self):
        self.scale_factor = 12
        self.net = model.enhance_net_nopool(self.scale_factor, conv_type='dsc').to(device)
        
        # 指定模型权重的路径
        self.weight_path = 'SGLE/weight/Epoch99.pth'
        self.net.load_state_dict(torch.load(self.weight_path, map_location=device))

    def inference(self, input_data):
        # 使用 image_from_input 处理输入数据
        data_lowlight = image_from_input(input_data)

        # 图像缩放为4的倍数分辨率
        data_lowlight = scale_image(data_lowlight, self.scale_factor, device) if self.scale_factor != 1 else data_lowlight

        # 运行模型推理
        start = time.time()
        enhanced_image, _ = self.net(data_lowlight)
        end_time = (time.time() - start)

        # 将tensor转为numpy
        enhanced_image_np = enhanced_image.squeeze().permute(1, 2, 0).cpu().detach().numpy() * 255.0
        enhanced_image_np = enhanced_image_np.astype(np.uint8)
        
        return enhanced_image_np, end_time

    def test(self):
        self.net.eval()

        # 指定要测试的单张图像的路径
        single_image_path = 'test.jpg'
        enhanced_image_np, elapsed_time = self.inference(single_image_path)

        # 保存增强后的图像
        cv2.imwrite('enhanced_test.jpg', cv2.cvtColor(enhanced_image_np, cv2.COLOR_RGB2BGR))

        print(f"处理时间: {elapsed_time} 秒")
        print("测试完成！")
