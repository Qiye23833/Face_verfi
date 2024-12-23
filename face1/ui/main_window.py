from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os

class MainWindow(QMainWindow):
    """
    主窗口类
    实现程序的图形用户界面，包含视频显示和控制功能
    """
    
    def __init__(self, detector, enhancer):
        """
        初始化主窗口
        
        Args:
            detector: FaceDetector实例，用于人体检测
            enhancer: ImageEnhancer实例，用于图像增强
        """
        super().__init__()
        self.detector = detector
        self.enhancer = enhancer
        self.camera_is_running = False
        self.current_image = None  # 存储当前显示的图像
        self.setup_ui()
        self.setup_camera()
        
    def setup_ui(self):
        """
        设置用户界面
        创建并布局界面组件
        """
        self.setWindowTitle('人脸考勤系统')
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧布局（视频显示）
        left_layout = QVBoxLayout()
        
        # 设置视频显示区域
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        left_layout.addWidget(self.video_label)
        
        # 添加按钮布局
        button_layout = QHBoxLayout()
        
        # 摄像头控制按钮
        self.camera_button = QPushButton('开启摄像头')
        self.camera_button.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.camera_button)
        
        # 加载图片按钮
        self.load_button = QPushButton('加载图片')
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)
        
        # 保存图片按钮
        self.save_button = QPushButton('保存图片')
        self.save_button.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_button)
        
        left_layout.addLayout(button_layout)
        main_layout.addLayout(left_layout)

        # 右侧信息显示区域
        info_layout = QVBoxLayout()
        self.info_label = QLabel('人员识别信息')
        info_layout.addWidget(self.info_label)
        main_layout.addLayout(info_layout)
        
    def setup_camera(self):
        """
        初始化摄像头和定时器
        设置视频捕获和画面更新机制
        """
        self.cap = None  # 初始化摄像头对象为None
        self.timer = QTimer()  # 创建定时器
        self.timer.timeout.connect(self.update_frame)  # 连接定时器到更新函数
        
    def toggle_camera(self):
        """
        切换摄像头状态
        开启或关闭摄像头
        """
        if not self.camera_is_running:
            # 开启摄像头
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_is_running = True
                self.timer.start(30)  # 开始定时更新
                self.camera_button.setText('关闭摄像头')
                self.video_label.setText('')  # 清除提示文本
            else:
                self.video_label.setText('无法打开摄像头')
        else:
            # 关闭摄像头
            self.stop_camera()
            
    def stop_camera(self):
        """
        停止摄像头
        释放摄像头资源并更新界面
        """
        self.timer.stop()  # 停止定时器
        if self.cap is not None:
            self.cap.release()  # 释放摄像头
            self.cap = None
        self.camera_is_running = False
        self.camera_button.setText('开启摄像头')
        self.video_label.setText('摄像头已关闭')

    def load_image(self):
        """
        加载图片功能
        打开文件对话框选择图片文件并显示
        """
        # 停止摄像头（如果正在运行）
        if self.camera_is_running:
            self.stop_camera()
            
        # 打开文件对话框
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )
        
        if file_name:
            # 读取图片
            frame = cv2.imread(file_name)
            if frame is not None:
                # 图像增强
                enhanced_frame = self.enhancer.enhance(frame)
                
                # 人体检测
                results = self.detector.detect(enhanced_frame)
                annotated_frame = self.detector.draw_detections(results)
                
                # 保存当前图像
                self.current_image = annotated_frame
                
                # 显示处理后的图像
                self.display_frame(annotated_frame)
            else:
                self.video_label.setText('无法加载图片')
                
    def save_image(self):
        """
        保存图片功能
        打开文件对话框选择保存位置并保存当前图像
        """
        if self.current_image is None:
            return
            
        # 打开保存文件对话框
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            "",
            "PNG图片 (*.png);;JPG图片 (*.jpg);;所有文件 (*.*)"
        )
        
        if file_name:
            # 确保文件名有正确的扩展名
            if not any(file_name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                file_name += '.png'
                
            # 保存图片
            cv2.imwrite(file_name, self.current_image)
            
    def update_frame(self):
        """
        更新视频帧
        捕获新的视频帧，进行处理并显示
        """
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 图像增强
                enhanced_frame = self.enhancer.enhance(frame)
                
                # 人体检测
                results = self.detector.detect(enhanced_frame)
                annotated_frame = self.detector.draw_detections(results)
                
                # 保存当前图像
                self.current_image = annotated_frame
                
                # 显示处理后的图像
                self.display_frame(annotated_frame)

    def display_frame(self, frame):
        """
        在界面上显示图像
        
        Args:
            frame: 要显示的图像帧（numpy数组格式，BGR颜色空间）
        """
        # 转换颜色空间并创建QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 显示图像
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
    def closeEvent(self, event):
        """
        窗口关闭事件处理
        
        Args:
            event: 关闭事件对象
        """
        self.stop_camera()  # 确保关闭摄像头
        event.accept()
        