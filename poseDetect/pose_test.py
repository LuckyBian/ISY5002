import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from time import time
from poseDetect.CustomResNet import ResNet101, ResNet18


# 加载已训练的模型
class CustomDeepModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomDeepModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),  # 输入通道数根据你的图像大小调整
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
# download_path = '/poseDetect/'
# torch.hub.set_dir(download_path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = CustomDeepModel(num_classes=3)
# model = ResNet18(num_classes=3).to(device)
# model.load_state_dict(torch.load('poseDetect/model_Resnet18_best.pth', map_location=torch.device('cpu')))
# model=model.to(device)
# model.eval()

# 预处理新的骨骼图
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 添加 batch 维度
    return image

# new_image_path = 'E:/NUS/ISY5002-main/ISY5002/static/skeleton.jpg'

# # 预处理图像
# input_image = preprocess_image(new_image_path)

# # 将输入图像移到GPU上（如果可用的话）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_image = input_image.to(device)

# # 使用模型进行预测
# with torch.no_grad():
#     start=time()
#     outputs = model(input_image)
#     end=time()

# # 获取预测结果

# _, predicted_class = torch.max(outputs, 1)


# # 打印预测结果
# print('0 for fall, 1 for normal, 2 for violent')
# print('Predicted Class:', predicted_class.item())
# print('time spending: {}'.format(end-start))
