import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO(model="weights/yolov5su.pt")

# 摄像头编号
camera_nu = 0

# 打开摄像头
cap = cv2.VideoCapture(camera_nu)

while cap.isOpened():
    # 获取图像
    res, frame = cap.read()
    # 如果读取成功
    if res:
        # 正向推理
        results = model(frame)
        # 绘制结果
        annotated_frame = results[0].plot()
        # 显示图像
        cv2.imshow(winname="YOLOV5", mat=annotated_frame)
        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break
# 释放连接
cap.release()
cv2.destroyAllWindows()