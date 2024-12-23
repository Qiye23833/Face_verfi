from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QScrollArea, QWidget, QPushButton, QGridLayout,
                            QLineEdit, QInputDialog, QMessageBox)
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
        
        # 设置主题颜色
        self.setStyleSheet("""
            QDialog {
                background-color: #F0F8FF;
            }
            QLabel {
                background-color: white;
                border: 1px solid #CCCCCC;
            }
            QPushButton {
                background-color: #E6F3FF;
                border: 1px solid #99CCFF;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #CCE6FF;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #CCCCCC;
                padding: 3px;
            }
        """)
        
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
            
            # ID输入框
            id_layout = QHBoxLayout()
            id_label = QLabel("ID:")
            id_edit = QLineEdit(str(id))
            id_edit.setReadOnly(True)  # ID只读
            id_layout.addWidget(id_label)
            id_layout.addWidget(id_edit)
            face_layout.addLayout(id_layout)
            
            # 姓名输入框
            name_layout = QHBoxLayout()
            name_label = QLabel("姓名:")
            name_edit = QLineEdit(name)
            name_edit.textChanged.connect(lambda text, id=id: self.update_name(id, text))
            name_layout.addWidget(name_label)
            name_layout.addWidget(name_edit)
            face_layout.addLayout(name_layout)
            
            # 按钮布局
            button_layout = QHBoxLayout()
            
            # 编辑ID按钮
            edit_id_btn = QPushButton("修改ID")
            edit_id_btn.clicked.connect(lambda _, old_id=id: self.edit_id(old_id))
            button_layout.addWidget(edit_id_btn)
            
            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.clicked.connect(lambda _, id=id: self.delete_face(id))
            button_layout.addWidget(delete_btn)
            
            face_layout.addLayout(button_layout)
            
            # 将人脸信息添加到网格布局
            col = row % 4
            row = row // 4
            self.content_layout.addWidget(face_widget, row, col)
    
    def update_name(self, face_id, new_name):
        """更新人脸姓名"""
        if new_name.strip():  # 确保名字不为空
            self.face_db.update_name(face_id, new_name)
    
    def edit_id(self, old_id):
        """编辑人脸ID"""
        new_id, ok = QInputDialog.getInt(
            self, '修改ID', '请输入新ID:', old_id, 1, 99999)
        if ok:
            if self.face_db.id_exists(new_id):
                QMessageBox.warning(self, '错误', 'ID已存在，请选择其他ID')
                return
            self.face_db.update_id(old_id, new_id)
            self.load_faces()  # 重新加载显示
    
    def delete_face(self, face_id):
        """删除指定的人脸信息"""
        reply = QMessageBox.question(
            self, '确认删除', 
            '确定要删除这条记录吗？',
            QMessageBox.Yes | QMessageBox.No)
            
        if reply == QMessageBox.Yes:
            self.face_db.delete_face(face_id)
            self.load_faces()  # 重新加载显示