from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2

class MainWindow(QMainWindow):
    def __init__(self, detector, enhancer):
        super().__init__()
        self.detector = detector
        self.enhancer = enhancer
        self.setup_ui()
        self.setup_camera()
        
    def setup_ui(self):
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
        
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 图像增强
            enhanced_frame = self.enhancer.enhance(frame)
            
            # 人脸检测
            results = self.detector.detect(enhanced_frame)
            annotated_frame = self.detector.draw_detections(results)
            
            # 显示图像
            self.display_frame(annotated_frame)
            
    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
    def closeEvent(self, event):
        self.cap.release()
        event.accept() 