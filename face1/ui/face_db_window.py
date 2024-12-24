from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QScrollArea, QWidget, QPushButton, QGridLayout,
                            QLineEdit, QInputDialog, QMessageBox, QFormLayout)
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
        faces = self.face_db.get_all_faces_with_info()  # 修改为获取完整信息
        
        # 清除现有内容
        for i in reversed(range(self.content_layout.count())): 
            self.content_layout.itemAt(i).widget().setParent(None)
            
        # 显示人脸信息
        for row, face_info in enumerate(faces):
            # 创建人脸信息容器
            face_widget = QWidget()
            face_layout = QVBoxLayout(face_widget)
            
            # 显示人脸图像
            if face_info['face_image'] is not None:
                img_array = np.frombuffer(face_info['face_image'], np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 创建图像标签并设置固定大小
                img_label = QLabel()
                img_label.setFixedSize(200, 200)
                img_label.setAlignment(Qt.AlignCenter)
                
                # 计算缩放比例
                h, w = img.shape[:2]
                scale = min(180/w, 180/h)
                new_size = (int(w * scale), int(h * scale))
                
                # 缩放图像
                resized_img = cv2.resize(img, new_size)
                
                # 创建白色背景
                background = np.full((200, 200, 3), 255, dtype=np.uint8)
                
                # 计算居中位置
                x_offset = (200 - new_size[0]) // 2
                y_offset = (200 - new_size[1]) // 2
                
                # 将图像放在背景中央
                background[y_offset:y_offset+new_size[1], 
                          x_offset:x_offset+new_size[0]] = resized_img
                
                # 转换为QPixmap并显示
                h, w = background.shape[:2]
                bytes_per_line = 3 * w
                q_img = QImage(background.data, w, h, bytes_per_line, QImage.Format_RGB888)
                img_label.setPixmap(QPixmap.fromImage(q_img))
                face_layout.addWidget(img_label)
            
            # 创建表单布局显示所有信息
            form_layout = QFormLayout()
            
            # ID显示
            id_edit = QLineEdit(str(face_info['id']))
            id_edit.setReadOnly(True)
            form_layout.addRow('ID:', id_edit)
            
            # 姓名输入框
            name_edit = QLineEdit(face_info['name'])
            name_edit.textChanged.connect(
                lambda text, id=face_info['id']: self.update_name(id, text))
            form_layout.addRow('姓名:', name_edit)
            
            # 性别显示
            gender_label = QLabel(face_info.get('gender', ''))
            form_layout.addRow('性别:', gender_label)
            
            # 岗位显示
            position_label = QLabel(face_info.get('position', ''))
            form_layout.addRow('岗位:', position_label)
            
            # 部门显示
            department_label = QLabel(face_info.get('department', ''))
            form_layout.addRow('部门:', department_label)
            
            # 人员类型显示
            type_label = QLabel(face_info.get('person_type', ''))
            form_layout.addRow('人员类型:', type_label)
            
            # 进驻时间显示
            entry_date_label = QLabel(face_info.get('entry_date', ''))
            form_layout.addRow('进驻时间:', entry_date_label)
            
            face_layout.addLayout(form_layout)
            
            # 按钮布局
            button_layout = QHBoxLayout()
            
            # 编辑ID按钮
            edit_id_btn = QPushButton("修改ID")
            edit_id_btn.clicked.connect(lambda _, old_id=face_info['id']: self.edit_id(old_id))
            button_layout.addWidget(edit_id_btn)
            
            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.clicked.connect(lambda _, id=face_info['id']: self.delete_face(id))
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