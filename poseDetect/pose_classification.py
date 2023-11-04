import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchviz import make_dot
from CustomResNet import ResNet101,ResNet50,ResNet34,ResNet18
from Baseline import VGG16,MobileNetV2,InceptionV3



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_data = datasets.ImageFolder(root='dataset/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

val_data = datasets.ImageFolder(root='dataset/val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

test_data = datasets.ImageFolder(root='dataset/test', transform=transform)
test_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimCNN, self).__init__()
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
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = ResNet18(num_classes=3).to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# summary(model,input_size=(1,224,224),device='cuda')


num_epochs = 500
best_val_accuracy = 0.0
train_losses = []
val_accuracies = []

model_name='Resnet18_500'
model_save_dir = 'save/best/'
os.makedirs(model_save_dir, exist_ok=True)


start=time.time()
for epoch in tqdm(range(num_epochs)):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader),
                                                                     loss.item()))

    # 在每个epoch结束后计算验证集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移到GPU上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    train_losses.append(loss.item())
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model_name = 'Resnet18_best'  # 模型文件名
        model_save_path = os.path.join(model_save_dir, f'model_{model_name}.pth')
        torch.save(model.state_dict(), model_save_path)

end=time.time()
print("training done!")
print('time spending: {}'.format(end-start))

# torch.save(model.state_dict(), os.path.join('save/', f'model_{model_name}.pth'))


model.load_state_dict(torch.load(model_save_path))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print('Test Accuracy: {:.2f}%'.format(test_accuracy))


# 绘制训练损失曲线和验证集准确率曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()