import cv2
from flask import Flask, Response, render_template
from SGLE.test2 import Tester
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from denoise.src.model import UDnCNN
from denoise.src import utils

app = Flask(__name__)

enhancer = Tester()


checkpoint = torch.load('denoise/model/checkpoint.pth.tar', map_location=torch.device('cpu'))
model = UDnCNN(6, 64)
model.load_state_dict(checkpoint['Net']) 
model.eval()  # 设置为评估模式
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 使用 OpenCV 来读取摄像头

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = frame.astype(np.uint8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, _ = enhancer.inference(frame_rgb)
        
        # 图像增强后的处理
            img_pil = Image.fromarray(processed_frame)
            transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # 降噪处理
            with torch.no_grad():
                denoised_img_tensor = model(img_tensor)
            denoised_img = (denoised_img_tensor * 0.5 + 0.5).clamp(0, 1).squeeze(0).cpu().numpy()
            denoised_img = np.moveaxis(denoised_img, 0, -1)
            denoised_img = (denoised_img * 255).astype(np.uint8)
        
        # 将处理后的图像转换回BGR格式以便显示
            denoised_img_bgr = cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR)
        
            ret, jpeg = cv2.imencode('.jpg', denoised_img_bgr)
            if ret:
                return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port=5001)
