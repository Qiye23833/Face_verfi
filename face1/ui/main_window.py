from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
from face1.utils.face_utils import FaceProcessor
from face1.utils.db_utils import FaceDatabase
from face1.ui.face_db_window import FaceDBWindow
import time
import numpy as np

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
        self.face_processor = FaceProcessor()
        self.face_db = FaceDatabase()
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
        
        # 图片检测按钮
        self.image_detect_button = QPushButton('图片人脸检测')
        self.image_detect_button.clicked.connect(self.detect_face_in_image)
        button_layout.addWidget(self.image_detect_button)
        
        # 摄像头检测按钮
        self.camera_detect_button = QPushButton('摄像头人脸检测')
        self.camera_detect_button.clicked.connect(self.toggle_camera_detection)
        button_layout.addWidget(self.camera_detect_button)
        
        # 图片注册按钮
        self.register_button = QPushButton('注册人脸')
        self.register_button.clicked.connect(self.register_face)
        button_layout.addWidget(self.register_button)
        
        # 查看数据库按钮
        self.view_db_button = QPushButton('查看数据库')
        self.view_db_button.clicked.connect(self.view_database)
        button_layout.addWidget(self.view_db_button)
        
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
        """停止摄像头"""
        self.timer.stop()  # 停止定时器
        if self.cap is not None:
            self.cap.release()  # 释放摄像头
            self.cap = None
        self.camera_is_running = False
        
        # 更新按钮文本
        self.camera_detect_button.setText('摄像头人脸检测')
        
        # 更新视频标签
        self.video_label.setText('摄像头已关闭')
        self.display_frame(np.zeros((600, 800, 3), dtype=np.uint8))  # 显示黑色画面

    def update_frame(self):
        """
        更新视频帧
        捕获新的视频帧，进行处理并显示
        """
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 处理帧进行人脸识别
                self.process_frame_for_recognition(frame)

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
        self.face_db.close()  # 关闭数据库连接
        event.accept()
        
    def register_face(self):
        """
        注册人脸功能
        包含图片加载、人脸检测、特征提取和保存功能
        """
        # 停止摄像头（如果正在运行）
        if self.camera_is_running:
            self.stop_camera()
        
        # 打开文件对话框选择图片
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择人脸图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )
        
        if not file_name:
            return
        
        # 读取图片
        frame = cv2.imread(file_name)
        if frame is None:
            self.info_label.setText('无法加载图片')
            return
        
        # 保存原始图像
        self.current_image = frame.copy()
        
        # 检测人脸
        faces = self.face_processor.detect_face(frame)
        if len(faces) == 0:
            self.info_label.setText('未检测到人脸')
            return
        
        # 使用置信度最高的人脸
        best_face = faces[0]
        
        # 提取人脸特征
        face_features = self.face_processor.extract_face_features(
            frame, best_face)
        
        # 对齐人脸
        aligned_face = self.face_processor.align_face(
            frame, best_face)
        
        # 获取姓名
        name, ok = QInputDialog.getText(self, '注册人脸', '请输入姓名:')
        if not ok or not name:
            return
        
        # 保存到数据库
        self.face_db.add_face(
            name, 
            face_features,
            cv2.imencode('.jpg', aligned_face)[1].tobytes()
        )
        
        # 在图像上标注注册的人脸
        x1, y1, x2, y2 = map(int, best_face[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 显示处理后的图像
        self.current_image = frame
        self.display_frame(frame)
        
        # 更新信息显示
        self.info_label.setText(f'已注册: {name}')
        
        # 询问是否保存标注后的图片
        reply = QMessageBox.question(
            self, 
            '保存图片', 
            '是否保存标注后的图片？',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 打开保存文件对话框
            save_name, _ = QFileDialog.getSaveFileName(
                self,
                "保存标注后的图片",
                f"{name}_registered.jpg",
                "JPG图片 (*.jpg);;PNG图片 (*.png);;所有文件 (*.*)"
            )
            
            if save_name:
                # 确保文件名有正确的扩展名
                if not any(save_name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                    save_name += '.jpg'
                
                # 保存图片
                cv2.imwrite(save_name, frame)
        
    def view_database(self):
        """打开数据库查看窗口"""
        db_window = FaceDBWindow(self.face_db, self)
        db_window.exec_()
        
    def detect_face_in_image(self):
        """图片人脸检测功能"""
        # 停止摄像头（如果正在运行）
        if self.camera_is_running:
            self.stop_camera()
        
        # 打开文件对话框选择图片
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )
        
        if not file_name:
            return
        
        # 读取图片
        frame = cv2.imread(file_name)
        if frame is None:
            self.info_label.setText('无法加载图片')
            return
        
        # 检测和识别人脸
        self.process_frame_for_recognition(frame)

    def toggle_camera_detection(self):
        """切换摄像头检测状态"""
        if not self.camera_is_running:
            # 开启摄像头
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_is_running = True
                self.timer.start(30)
                self.camera_detect_button.setText('停止检测')
                self.video_label.setText('')
            else:
                self.video_label.setText('无法打开摄像头')
        else:
            self.stop_camera()
            self.camera_detect_button.setText('摄像头人脸检测')

    def process_frame_for_recognition(self, frame):
        """处理帧进行人脸识别"""
        try:
            # 保存原始图像
            self.current_image = frame.copy()
            
            # 检测人脸
            faces = self.face_processor.detect_face(frame)
            if len(faces) == 0:
                self.info_label.setText('未检测到人脸')
                self.display_frame(frame)
                return
            
            # 处理每个检测到的人脸
            recognition_results = []
            for face_box in faces:
                try:
                    # 提取人脸特征
                    face_features = self.face_processor.extract_face_features(frame, face_box)
                    if face_features is None:
                        continue
                    
                    # 与数据库中的人脸进行匹配
                    match_result = self.face_db.match_face(face_features)
                    
                    # 在图像上标注结果
                    x1, y1, x2, y2 = map(int, face_box[:4])
                    if match_result:
                        # 匹配成功，显示姓名和相似度
                        id, name, similarity = match_result
                        label = f"{name} ({similarity:.2f})"
                        color = (0, 255, 0)  # 绿色
                        recognition_results.append(f"{name}: {similarity:.2f}")
                    else:
                        # 匹配失败
                        label = "Unknown"
                        color = (0, 0, 255)  # 红色
                        recognition_results.append("未识别")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue
            
            # 显示处理后的图像
            self.display_frame(frame)
            
            # 更新识别结果显示
            if recognition_results:
                self.info_label.setText("识别结果:\n" + "\n".join(recognition_results))
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
        
        