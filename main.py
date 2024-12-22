from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
import cv2
import torch

class FaceAttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('人脸考勤系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 左侧视频显示区域
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        layout.addWidget(self.video_label)
        
        # 右侧信息显示区域
        info_layout = QVBoxLayout()
        self.info_label = QLabel('人员信息')
        info_layout.addWidget(self.info_label)
        layout.addLayout(info_layout)
        
        # 初始化YOLOv5模型
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5s.pt')
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        # 设置定时器更新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms刷新一次

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 图像增强处理
            enhanced_frame = self.enhance_image(frame)
            
            # 人脸检测
            results = self.model(enhanced_frame)
            
            # 在图像上标注检测结果
            annotated_frame = results.render()[0]
            
            # 转换图像格式用于显示
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def enhance_image(self, frame):
        # 图像增强处理
        # 这里可以添加各种图像增强方法，如：
        # 1. 直方图均衡化
        # 2. 亮度调整
        # 3. 对比度增强等
        enhanced = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceAttendanceSystem()
    window.show()
    sys.exit(app.exec_()) 