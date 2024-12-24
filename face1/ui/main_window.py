import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QInputDialog, 
                            QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from face1.utils.face_utils import FaceProcessor
from face1.utils.db_utils import FaceDatabase
from face1.ui.face_db_window import FaceDBWindow
from face1.ui.register_dialog import RegisterDialog
import time
from PIL import Image, ImageDraw, ImageFont

class MainWindow(QMainWindow):
    """
    主窗口类
    实现程序的图形用户界面，包含视频显示和控制功能
    """
    
    def __init__(self):
        """
        初始化主窗口
        """
        super().__init__()
        self.camera_is_running = False
        self.current_image = None  # 存储当前显示的图像
        self.face_processor = FaceProcessor()
        self.face_db = FaceDatabase()
        self.setup_ui()
        self.setup_camera()
        
        # 加载中文字体
        try:
            self.font = ImageFont.truetype("simhei.ttf", 40)  # 使用黑体
        except:
            try:
                self.font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 40)  # 尝试使用 Noto 字体
            except:
                print("未找到合适的中文字体，将使用默认字体")
                self.font = None
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle('人脸考勤系统')
        self.setGeometry(100, 100, 1400, 900)  # 增大窗口尺寸
        
        # 设置主题颜色和字体样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F0F8FF;
            }
            QLabel {
                background-color: white;
                border: 1px solid #CCCCCC;
                font-size: 14px;
            }
            QPushButton {
                background-color: #E6F3FF;
                border: 1px solid #99CCFF;
                border-radius: 4px;
                padding: 8px;
                min-width: 100px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #CCE6FF;
            }
            #infoLabel {
                font-size: 16px;
                font-family: Arial;
                padding: 10px;
                line-height: 1.5;
            }
        """)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)  # 增加组件间距

        # 左侧布局（视频显示）
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)  # 增加垂直间距
        
        # 设置视频显示区域
        self.video_label = QLabel()
        self.video_label.setFixedSize(900, 700)  # 增大视频显示区域
        self.video_label.setStyleSheet("border: 2px solid #99CCFF;")
        left_layout.addWidget(self.video_label)
        
        # 添加按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)  # 增加按钮间距
        
        # 图片检测按钮
        self.image_detect_button = QPushButton('图片人脸检测')
        self.image_detect_button.clicked.connect(self.detect_face_in_image)
        button_layout.addWidget(self.image_detect_button)
        
        # 视频检测按钮
        self.video_detect_button = QPushButton('视频人脸检测')
        self.video_detect_button.clicked.connect(self.detect_face_in_video)
        button_layout.addWidget(self.video_detect_button)
        
        # 摄像头检测按钮
        self.camera_detect_button = QPushButton('摄像头人脸检测')
        self.camera_detect_button.clicked.connect(self.toggle_camera_detection)
        button_layout.addWidget(self.camera_detect_button)
        
        # 注册人脸按钮
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
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)  # 增加垂直间距
        
        # 人脸图像显示
        self.face_image_label = QLabel('人脸图像')
        self.face_image_label.setFixedSize(400, 300)  # 增大人脸图像显示区域
        self.face_image_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.face_image_label)
        
        # 添加一些间距
        right_layout.addSpacing(20)
        
        # ID和姓名信息显示
        self.info_label = QLabel('等待识别...')
        self.info_label.setObjectName("infoLabel")  # 设置对象名以应用特定样式
        self.info_label.setFixedSize(400, 420)  # 增大信息显示区域
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.setWordWrap(True)  # 允许文本换行
        self.info_label.setMargin(10)  # 添加内边距
        right_layout.addWidget(self.info_label)
        
        # 添加弹性空间
        right_layout.addStretch()
        
        main_layout.addLayout(right_layout)
        
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
        在界面上显示图像，保持比例并完整显示
        
        Args:
            frame: 要显示的图像帧（numpy数组格式，BGR颜色空间）
        """
        if frame is None:
            return
        
        # 获取标签和图像的尺寸
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        img_height, img_width = frame.shape[:2]
        
        # 计算缩放比例
        width_ratio = label_width / img_width
        height_ratio = label_height / img_height
        scale = min(width_ratio, height_ratio)
        
        # 计算缩放后的尺寸
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # 缩放图像
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 创建一个黑色背景图像，尺寸与标签相同
        background = np.zeros((label_height, label_width, 3), dtype=np.uint8)
        
        # 计算图像在标签中的位置（居中显示）
        x_offset = (label_width - new_width) // 2
        y_offset = (label_height - new_height) // 2
        
        # 将缩放后的图像放在背景中央
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        
        # 转换颜色空间创建QImage
        rgb_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
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
        if self.camera_is_running:
            if hasattr(self.cap, 'get') and self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= 0:
                self.stop_video()  # 如果是视频，调用stop_video
            else:
                self.stop_camera()  # 如果是摄像头，调用stop_camera
        self.face_db.close()
        event.accept()
        
    def register_face(self):
        """注册人脸功能"""
        try:
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
            
            # 检查文件是否存在
            if not os.path.exists(file_name):
                QMessageBox.warning(self, '错误', '文件不存在')
                return
            
            # 检查文件是否可读
            if not os.access(file_name, os.R_OK):
                QMessageBox.warning(self, '错误', '文件无法读取')
                return
            
            # 读取图片
            frame = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                QMessageBox.warning(self, '错误', '无法加载图片，请确认文件格式正确')
                return
            
            # 保存并显示原始图像
            self.current_image = frame.copy()
            self.display_frame(frame)
            
            # 检测人脸
            faces = self.face_processor.detect_face(frame)
            if len(faces) == 0:
                self.info_label.setText('未检测到人脸')
                return
            elif len(faces) > 1:
                self.info_label.setText('检测到多个人脸，请确保图片中只有一个脸')
                return
            
            # 使用检测到的人脸
            face_box = faces[0]
            
            try:
                # 提取人脸特征
                face_features = self.face_processor.extract_face_features(frame, face_box)
                if face_features is None:
                    self.info_label.setText('无法提取人脸特征')
                    return
                
                # 对齐人脸
                aligned_face = self.face_processor.align_face(frame, face_box)
                if aligned_face is None:
                    self.info_label.setText('人脸对齐失败')
                    return
                    
                # 确保对齐后的人脸是RGB格式
                if len(aligned_face.shape) == 3:
                    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                
                # 在左侧大屏幕上标注人脸位置
                x1, y1, x2, y2 = map(int, face_box[:4])
                frame_with_box = frame.copy()
                cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.display_frame(frame_with_box)
                
                # 在右侧显示对齐后的人脸
                self.display_face_image(aligned_face)
                
                # 显示提示信息
                self.info_label.setText('人脸检测成功\n请输入ID和姓名进行注册')
                
                # 等待用户确认是否继续注册
                reply = QMessageBox.question(
                    self,
                    '确认注册',
                    '是否要注册该人脸？',
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return
                
                # 打开注册对话框
                dialog = RegisterDialog(self.face_db, aligned_face, face_features, self)
                if dialog.exec_() == QDialog.Accepted and dialog.result is not None:
                    try:
                        # 保存到数据库
                        face_image_bytes = cv2.imencode('.jpg', aligned_face)[1].tobytes()
                        self.face_db.add_face_with_info(
                            dialog.result,
                            face_features,
                            face_image_bytes
                        )
                        
                        # 在图像上添加ID和姓名标注
                        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_with_box, 
                                  f"{dialog.result['name']} (ID: {dialog.result['id']})", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # 显示最终的标注图像
                        self.display_frame(frame_with_box)
                        
                        # 显示注册成功信息
                        self.info_label.setText(f"注册成功\nID: {dialog.result['id']}\n"
                                              f"姓名: {dialog.result['name']}")
                        
                    except Exception as e:
                        print(f"Database error: {str(e)}")
                        QMessageBox.critical(self, '错误', '保存到数据库时出错')
                        return
                        
            except Exception as e:
                print(f"Face processing error: {str(e)}")
                QMessageBox.critical(self, '错误', '人脸处理过程中出错')
                return
                
        except Exception as e:
            print(f"Registration error: {str(e)}")
            QMessageBox.critical(self, '错误', '注册过程中出错')
            return
        
    def view_database(self):
        """打开数据库查看窗口"""
        db_window = FaceDBWindow(self.face_db, self)
        db_window.exec_()
        
    def detect_face_in_image(self):
        """图片人脸检测功能"""
        try:
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
            
            # 检查文件是否存在
            if not os.path.exists(file_name):
                QMessageBox.warning(self, '错误', '文件不存在')
                return
            
            # 检查文件是否可读
            if not os.access(file_name, os.R_OK):
                QMessageBox.warning(self, '错误', '文件无法读取')
                return
            
            # 读取图片
            frame = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                QMessageBox.warning(self, '错误', '无法加载图片，请确认文件格式正确')
                return
            
            # 检测和识别人脸
            self.process_frame_for_recognition(frame)
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            QMessageBox.warning(self, '错误', f'加载图片时出错：{str(e)}')

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

    def cv2_add_chinese_text(self, img, text, position, color):
        """使用PIL添加中文文字到图片上"""
        if self.font is None:
            # 如果没有找到中文字体，使用OpenCV添加英文
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            return img
            
        # 转换图片为PIL格式
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 创建绘图对象
        draw = ImageDraw.Draw(pil_img)
        
        # 添加文字
        draw.text(position, text, font=self.font, fill=color[::-1])  # PIL使用RGB顺序
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def process_frame_for_recognition(self, frame):
        """处理帧进行人脸识别"""
        try:
            # 保存原始图像
            self.current_image = frame.copy()
            
            # 检测人脸
            faces = self.face_processor.detect_face(frame)
            if len(faces) == 0:
                self.info_label.setText('人脸检测失败')
                self.display_frame(frame)
                return
            
            # 如果检测到多个人脸，选择面积最大的
            if len(faces) > 1:
                # 计算每个人脸框的面积
                face_areas = []
                for face_box in faces:
                    x1, y1, x2, y2 = map(int, face_box[:4])
                    area = (x2 - x1) * (y2 - y1)
                    face_areas.append(area)
                
                # 选择面积最大的人脸
                largest_face_idx = face_areas.index(max(face_areas))
                faces = [faces[largest_face_idx]]
            
            # 处理选中的人脸
            face_box = faces[0]
            try:
                # 提取人脸特征
                face_features = self.face_processor.extract_face_features(frame, face_box)
                if face_features is None:
                    return
                
                # 与数据库中的人脸进行匹配
                match_result = self.face_db.match_face(face_features)
                
                # 在图像上标注结果
                x1, y1, x2, y2 = map(int, face_box[:4])
                if match_result:
                    # 匹配成功，显示姓名和相似度
                    id, name, similarity = match_result
                    label = f"{name} ({similarity:.2f})"
                    color = (0, 255, 0)  # 绿色
                    
                    # 获取完整的人脸信息
                    face_info = self.face_db.get_face_info(id)
                    if face_info:
                        info_text = (
                            f"<b>识别结果：</b><br><br>"
                            f"<b>ID:</b> {face_info['id']}<br><br>"
                            f"<b>姓名:</b> {face_info['name']}<br><br>"
                            f"<b>性别:</b> {face_info['gender']}<br><br>"
                            f"<b>岗位:</b> {face_info['position']}<br><br>"
                            f"<b>部门:</b> {face_info['department']}<br><br>"
                            f"<b>人员类型:</b> {face_info['person_type']}<br><br>"
                            f"<b>进驻时间:</b> {face_info['entry_date']}<br><br>"
                            f"<b>相似度:</b> {similarity:.2f}"
                        )
                        
                        # 显示匹配到的人脸图像
                        face_img = self.face_processor.align_face(frame, face_box)
                        if face_img is not None:
                            self.display_face_image(face_img)
                        
                        # 显示完整信息
                        self.info_label.setText(info_text)
                else:
                    # 匹配失败
                    label = "未识别"
                    color = (0, 0, 255)  # 红色
                    self.info_label.setText("未识别")
                
                # 在图像上标注结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 使用新方法添加中文文字
                frame = self.cv2_add_chinese_text(
                    frame, 
                    label,
                    (x1, max(y1-40, 0)),  # 调整文字位置，避免超出图像边界
                    color
                )
                        
            except Exception as e:
                print(f"Error processing face: {str(e)}")
            
            # 显示处理后的图像
            self.display_frame(frame)
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
        
    def display_face_image(self, face_img):
        """在右侧显示人脸图像"""
        h, w = face_img.shape[:2]
        bytes_per_line = 3 * w
        q_img = QImage(face_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.face_image_label.setPixmap(scaled_pixmap)
        
    def detect_face_in_video(self):
        """视频人脸检测功能"""
        # 停止摄像头（如果正在运行）
        if self.camera_is_running:
            self.stop_camera()
        
        # 打开文件对话框选择视频
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)"
        )
        
        if not file_name:
            return
        
        # 打开视频文件
        self.cap = cv2.VideoCapture(file_name)
        if not self.cap.isOpened():
            self.info_label.setText('无法打开视频文件')
            return
        
        # 更新UI状态
        self.camera_is_running = True
        self.video_detect_button.setText('停止检测')
        self.video_label.setText('')
        
        # 获取视频信息
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # 如果无法获取fps，使用默认值
        
        # 设置定时器间隔
        interval = int(1000 / fps)  # 转换为毫秒
        self.timer.start(interval)
        
        # 清空右侧显示
        self.face_image_label.clear()
        self.info_label.setText('开始视频检测...')

    def stop_video(self):
        """停止视频播放"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_is_running = False
        self.video_detect_button.setText('视频人脸检测')
        self.video_label.setText('视频已停止')
        self.display_frame(np.zeros((600, 800, 3), dtype=np.uint8))
        
        