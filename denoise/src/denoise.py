import torch
import torchvision as tv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import UDnCNN
import torchvision.transforms as transforms

def myimshow(image, ax=plt):
    image = image.to('cpu').detach().numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

def image_preprocess(img_path):
    img = Image.open(img_path).convert('RGB')  
    transform = tv.transforms.Compose([
        tv.transforms.Resize(300),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    img = transform(img)
    return img

def denoise_image(model, img_tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        denoised_img_tensor = model(img_tensor)
        
    return denoised_img_tensor[0]

# 加载模型
checkpoint = torch.load('../model/checkpoint.pth.tar', map_location=torch.device('cpu'))
model = UDnCNN(6,64)

# 加载模型状态
model.load_state_dict(checkpoint['Net']) 

# 设置图像路径
img_path = 'test.jpeg' 

# 处理图像
img_tensor = image_preprocess(img_path)

# 进行降噪
denoised_image_tensor = denoise_image(model, img_tensor)

# 保存降噪后的图像
to_pil = transforms.ToPILImage()
denoised_image = to_pil((denoised_image_tensor * 0.5 + 0.5).clamp(0, 1).cpu())
denoised_image.save('denoised_image.png')
