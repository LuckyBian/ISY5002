import cv2
from flask import Flask, Response, render_template, request, jsonify, send_file
from SGLE.test2 import Tester
import torch
import numpy as np
import base64
import torchvision.transforms as transforms
from PIL import Image
from denoise.src.model import UDnCNN
from denoise.src import utils
import tempfile
import os
import logging
from poseDetect.skeletion.test import process_image
from poseDetect.pose_test import preprocess_image
from poseDetect.CustomResNet import ResNet101, ResNet18, ResNet34

app = Flask(__name__)

enhancer = Tester()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

checkpoint = torch.load('denoise/model/checkpoint.pth.tar', map_location=torch.device('cpu'))
model = UDnCNN(6, 64)
model.load_state_dict(checkpoint['Net']) 
model.eval() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_available = self.cap.isOpened()  # Check if the camera is available
        if not self.camera_available:
            print("Camera not found or cannot be opened")


    def get_frame(self):
        if self.camera_available:
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = frame.astype(np.uint8)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame, _ = enhancer.inference(frame_rgb)
                    
                    img_pil = Image.fromarray(processed_frame)
                    transform = transforms.Compose([
                        transforms.Resize((300, 300)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)


                    with torch.no_grad():
                        denoised_img_tensor = model(img_tensor)
                    denoised_img = (denoised_img_tensor * 0.5 + 0.5).clamp(0, 1).squeeze(0).cpu().numpy()
                    denoised_img = np.moveaxis(denoised_img, 0, -1)
                    denoised_img = (denoised_img * 255).astype(np.uint8)
        

                    denoised_img_bgr = cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR)
        
                    ret, jpeg = cv2.imencode('.jpg', denoised_img_bgr)
                    if ret:
                        return jpeg.tobytes()
            except Exception as e:
                # If an exception occurs, you can handle the error here or simply return None
                pass

        # If camera is not available or an error occurred, return an error frame
        error_message = "Camera not available" if not self.camera_available else "Error occurred"
        error_frame = np.zeros((300, 500, 3), dtype=np.uint8)  # Increase the frame size
        cv2.putText(error_frame, error_message, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Adjust text position
        ret, jpeg = cv2.imencode('.jpg', error_frame)
        if ret:
            return jpeg.tobytes()

def process_uploaded_video(input_video, output_video_path):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(input_video.read())
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'X264'), 30, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame.astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, _ = enhancer.inference(frame_rgb)

        # img_pil = Image.fromarray(processed_frame)
        # transform = transforms.Compose([
        #     transforms.Resize((300, 300)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])
        # img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # with torch.no_grad():
        #     denoised_img_tensor = model(img_tensor)
        # denoised_img = (denoised_img_tensor * 0.5 + 0.5).clamp(0, 1).squeeze(0).cpu().numpy()
        # denoised_img = np.moveaxis(denoised_img, 0, -1)
        # denoised_img = (denoised_img * 255).astype(np.uint8)
        
        # processed_frame = cv2.bitwise_not(frame)

        out.write(processed_frame)

    cap.release()
    out.release()

    os.remove(temp_file_path)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break


@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect', methods=['POST'])
def detect():
    picture = request.files.get('image')
    
    try:
        download_path = '/poseDetect/'
        torch.hub.set_dir(download_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = CustomDeepModel(num_classes=3)
        model = ResNet34(num_classes=3).to(device)
        model.load_state_dict(torch.load('poseDetect/model_Resnet34mini_best.pth', map_location=torch.device('cpu')))
        model=model.to(device)
        model.eval()

        picture_data = picture.read()
        picture_array = cv2.imdecode(np.frombuffer(picture_data, np.uint8), -1)
        

        skeleton_image = process_image(picture_array)
        
 
        skeleton_image_path = 'static/skeleton.jpg'
        cv2.imwrite(skeleton_image_path, skeleton_image)

        input_image = preprocess_image(skeleton_image_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_image = input_image.to(device)

        with torch.no_grad():
            outputs = model(input_image)

        _, predicted_class = torch.max(outputs, 1)

        return jsonify({'skeleton_image_path': skeleton_image_path, 'predicted_class': predicted_class.item()})
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500


@app.route('/process', methods=['POST'])
def process():

    picture = request.files.get('picture')
    

    try:
        picture_data = picture.read()
        # picture_array = cv2.imdecode(np.frombuffer(picture_data, np.uint8), -1)
        # picture_gray = cv2.cvtColor(picture_array, cv2.COLOR_BGR2GRAY)
        # picture_data = picture_data.astype(np.uint8)
        picture_array = cv2.imdecode(np.frombuffer(picture_data, np.uint8), -1)
        picture_array_rgb = cv2.cvtColor(picture_array, cv2.COLOR_BGR2RGB)

        original_height, original_width, _ = picture_array.shape

        target_height = 650
        target_width = int(original_width * (target_height / original_height))

        processed_frame, _ = enhancer.inference(picture_array_rgb)
                    
        img_pil = Image.fromarray(processed_frame)
        transform = transforms.Compose([
            transforms.Resize((original_height, original_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            denoised_img_tensor = model(img_tensor)
        denoised_img = (denoised_img_tensor * 0.5 + 0.5).clamp(0, 1).squeeze(0).cpu().numpy()
        denoised_img = np.moveaxis(denoised_img, 0, -1)
        denoised_img = (denoised_img * 255).astype(np.uint8)
        
        denoised_img_bgr = cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR)

        image_width = denoised_img_bgr.shape[1]
        image_height = denoised_img_bgr.shape[0]

        picture_preview = 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', denoised_img_bgr)[1]).decode()

        return jsonify({
            'picture_preview': picture_preview,
            'image_width': image_width, 
            'image_height': image_height 
        })
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500


@app.route('/process_video', methods=['POST'])
def process_video():
    video = request.files.get('video')

    if not video:
        return jsonify({'error': 'Video file not provided'}), 400

    output_video_path = os.path.join('static', 'processed_video.mkv')

    process_uploaded_video(video, output_video_path)

    logging.info(f'Processed video file path: {output_video_path}')

    return jsonify({"processed_video_path": output_video_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port=5001,threaded=True)
