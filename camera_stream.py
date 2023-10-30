import cv2
from flask import Flask, Response, render_template
from SGLE.test2 import Tester
import torch
import numpy as np

app = Flask(__name__)

enhancer = Tester()

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 使用 OpenCV 来读取摄像头

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = frame.astype(np.uint8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, _ = enhancer.inference(frame_rgb)
            denoised_frame = cv2.medianBlur(processed_frame, ksize=5)
        
            ret, jpeg = cv2.imencode('.jpg', denoised_frame)
            #ret, jpeg = cv2.imencode('.jpg', processed_frame)
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
