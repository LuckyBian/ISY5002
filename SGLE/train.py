import os
import Myloss
import dataloader
from modeling import model
import torch.optim
from modeling.fpn import *
from option import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置CUDA设备，只使用第一个GPU
device = get_device()  # 获取设备，可能是CPU或者GPU

class Trainer():
    def __init__(self):
        self.scale_factor = args.scale_factor  # 设置缩放因子

        self.net = model.enhance_net_nopool(self.scale_factor, conv_type=args.conv_type).to(device)  # 初始化增强网络
        
        self.seg = fpn(args.num_of_SegClass).to(device)  # 初始化分割网络


        self.seg_criterion = FocalLoss(gamma=2).to(device)  # 设置分割网络的损失函数

        self.train_dataset = dataloader.lowlight_loader(args.lowlight_images_path)  # 加载低光照图像数据集

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,  # 创建数据加载器
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True)
        
        self.L_color = Myloss.L_color()  # 初始化颜色损失
        self.L_spa = Myloss.L_spa8(patch_size=args.patch_size)  # 初始化空间一致性损失
        self.L_exp = Myloss.L_exp(16)  # 初始化曝光损失
        self.L_TV = Myloss.L_TV()  # 初始化总变差损失

        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 设置优化器

        self.num_epochs = args.num_epochs  # 设置训练周期
        self.E = args.exp_level  # 设置曝光级别
        self.grad_clip_norm = args.grad_clip_norm  # 设置梯度裁剪的阈值
        self.display_iter = args.display_iter  # 设置每多少次迭代显示一次损失
        self.snapshot_iter = args.snapshot_iter  # 设置每多少次迭代保存一次模型
        self.snapshots_folder = args.snapshots_folder  # 设置模型保存的文件夹

        if args.load_pretrain == True:  # 如果需要加载预训练模型
            self.net.load_state_dict(torch.load(args.pretrain_dir, map_location=device))  # 加载预训练模型
            print("weight is OK")



    # 图像分割的损失
    def get_seg_loss(self, enhanced_image):

        seg_input = enhanced_image.to(device)  # 将增强后的图像移到设备上
        seg_output = self.seg(seg_input).to(device)  # 获取分割结果

        target = (get_NoGT_target(seg_output)).data.to(device)  # 获取无标签的目标

        seg_loss = self.seg_criterion(seg_output, target)  # 计算分割损失

        return seg_loss

     # 图像增强的损失   
    def get_loss(self, A, enhanced_image, img_lowlight, E):
        Loss_TV = 1600 * self.L_TV(A)  # 计算总变差损失
        loss_spa = torch.mean(self.L_spa(enhanced_image, img_lowlight))  # 计算空间一致性损失
        loss_col = 5 * torch.mean(self.L_color(enhanced_image))  # 计算颜色损失
        loss_exp = 10 * torch.mean(self.L_exp(enhanced_image, E))  # 计算曝光损失
        loss_seg = self.get_seg_loss(enhanced_image)  # 计算分割损失

        loss = Loss_TV + loss_spa + loss_col + loss_exp + 0.1 * loss_seg  # 总损失

        return loss

def train(self):
    self.net.train()  # 设置为训练模式

    for epoch in range(self.num_epochs):  # 遍历所有训练周期

        for iteration, img_lowlight in enumerate(self.train_loader):  # 遍历数据集

            img_lowlight = img_lowlight.to(device)  # 将数据移到设备上

            enhanced_image, A = self.net(img_lowlight)  # 获取增强图像和中间变量A

            loss = self.get_loss(A, enhanced_image, img_lowlight, self.E)  # 计算总损失

            self.optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm(self.net.parameters(), self.grad_clip_norm)  # 裁剪梯度
            self.optimizer.step()  # 更新权重

            if ((iteration + 1) % self.display_iter) == 0:  # 显示损失
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % self.snapshot_iter) == 0:  # 保存模型
                torch.save(self.net.state_dict(), self.snapshots_folder + 'model.pth')


if __name__ == "__main__":
    t = Trainer()  # 创建训练器对象
    t.train()  # 开始训练










