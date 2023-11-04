import cv2 as cv
import numpy as np
import os

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


def process_image(input, thr=0.2, width=368, height=368):
    current_dir = os.path.dirname(__file__)

# 构建模型文件的相对路径
    model_path = os.path.join(current_dir, "graph_opt.pb")

# 使用相对路径加载模型
    net = cv.dnn.readNetFromTensorflow(model_path)
    # net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
    
    frame = input  # 直接使用传入的图像数据
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert (len(BODY_PARTS) == out.shape[1])

    skeleton_image = np.zeros_like(frame)  # 创建一个与输入图像相同大小的全黑图像

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        # print(f"HeatMap shape: {heatMap.shape}")
        # print(heatMap)
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(skeleton_image, points[idFrom], points[idTo], (255, 255, 255),  8)

    return skeleton_image



# 示例用法
# if __name__ == "__main__":
#     input_path = r'L:\test\skeletion\DSC09546.JPG'
#     skeleton_image = process_image(input_path)
#     cv.imwrite('skeleton.jpg', skeleton_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
