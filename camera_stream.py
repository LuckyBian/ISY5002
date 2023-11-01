import cv2
from flask import Flask, Response, render_template, request, jsonify
from SGLE.test2 import Tester
import torch
import numpy as np
import base64

app = Flask(__name__)

enhancer = Tester()

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
                    ret, jpeg = cv2.imencode('.jpg', processed_frame)
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

# 这个路由用于接收并处理上传的图片和视频
@app.route('/process', methods=['POST'])
def process():
    picture = request.files.get('picture')
    video = request.files.get('video')

    if picture:
        # 处理上传的图片
        picture_data = picture.read()
        picture_array = cv2.imdecode(np.frombuffer(picture_data, np.uint8), -1)
        picture_gray = cv2.cvtColor(picture_array, cv2.COLOR_BGR2GRAY)

        # 将图片的字节序列转换为Base64编码的字符串
        picture_preview = 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', picture_gray)[1]).decode()

        # 返回处理后的图片
        return jsonify({
            'picture_preview': picture_preview,
            'video_preview': None  # 未上传视频时返回空
        })
    elif video:
        # 处理上传的视频
        video_data = video.read()
        video_array = cv2.imdecode(np.frombuffer(video_data, np.uint8), -1)
        video_gray = cv2.cvtColor(video_array, cv2.COLOR_BGR2GRAY)

        # 将视频的字节序列转换为Base64编码的字符串
        video_preview = 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', video_gray)[1]).decode()

        # 返回处理后的视频
        return jsonify({
            'picture_preview': None,  # 未上传图片时返回空
            'video_preview': video_preview,
        })
    else:
        return 'Invalid file data'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port=5001)
