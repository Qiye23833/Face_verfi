from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QScrollArea, QWidget, QPushButton, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

class FaceDBWindow(QDialog):
    """人脸数据库查看窗口"""
    
    def __init__(self, face_db, parent=None):
        super().__init__(parent)
        self.face_db = face_db
        self.setup_ui()
        self.load_faces()
        
    def setup_ui(self):
        """设置界面"""
        self.setWindowTitle('人脸数据库')
        self.setGeometry(200, 200, 800, 600)
        
        # 创建主布局
        layout = QVBoxLayout(self)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # 创建滚动区域的内容窗口
        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        scroll.setWidget(self.content_widget)
        
    def load_faces(self):
        """加载并显示所有人脸信息"""
        # 获取所有人脸信息
        faces = self.face_db.get_all_faces_with_images()
        
        # 清除现有内容
        for i in reversed(range(self.content_layout.count())): 
            self.content_layout.itemAt(i).widget().setParent(None)
            
        # 显示人脸信息
        for row, (id, name, feature_vector, face_image) in enumerate(faces):
            # 创建人脸信息容器
            face_widget = QWidget()
            face_layout = QVBoxLayout(face_widget)
            
            # 显示人脸图像
            if face_image is not None:
                img_array = np.frombuffer(face_image, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                img_label = QLabel()
                img_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
                face_layout.addWidget(img_label)
            
            # 显示姓名和ID
            info_label = QLabel(f"ID: {id}\n姓名: {name}")
            info_label.setAlignment(Qt.AlignCenter)
            face_layout.addWidget(info_label)
            
            # 添加删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.clicked.connect(lambda checked, id=id: self.delete_face(id))
            face_layout.addWidget(delete_btn)
            
            # 将人脸信息添加到网格布局
            col = row % 4
            row = row // 4
            self.content_layout.addWidget(face_widget, row, col)
            
    def delete_face(self, face_id):
        """删除指定的人脸信息"""
        self.face_db.delete_face(face_id)
        self.load_faces()  # 重新加载显示 