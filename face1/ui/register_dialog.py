from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QComboBox, QDateEdit, QPushButton,
                            QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

class RegisterDialog(QDialog):
    """人脸注册对话框"""
    
    def __init__(self, face_db, face_image, face_features, parent=None):
        super().__init__(parent)
        self.face_db = face_db
        self.face_image = face_image
        self.face_features = face_features
        self.result = None
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle('注册人脸信息')
        self.setFixedWidth(500)
        
        # 设置主题颜色
        self.setStyleSheet("""
            QDialog {
                background-color: #F0F8FF;
            }
            QLabel {
                background-color: white;
                border: 1px solid #CCCCCC;
                padding: 5px;
            }
            QLineEdit, QComboBox, QDateEdit {
                background-color: white;
                border: 1px solid #99CCFF;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QPushButton {
                background-color: #E6F3FF;
                border: 1px solid #99CCFF;
                border-radius: 4px;
                padding: 8px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #CCE6FF;
            }
        """)
        
        # 创建主布局
        layout = QVBoxLayout(self)
        
        # 显示人脸图像
        image_label = QLabel()
        image_label.setFixedSize(200, 200)
        image_label.setAlignment(Qt.AlignCenter)
        
        # 转换并显示图像
        h, w = self.face_image.shape[:2]
        bytes_per_line = 3 * w
        q_img = QImage(self.face_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        
        layout.addWidget(image_label, alignment=Qt.AlignCenter)
        
        # 创建表单布局
        form_layout = QFormLayout()
        
        # ID输入框
        self.id_edit = QLineEdit()
        form_layout.addRow('ID:', self.id_edit)
        
        # 姓名输入框
        self.name_edit = QLineEdit()
        form_layout.addRow('姓名:', self.name_edit)
        
        # 性别选择
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(['男', '女'])
        form_layout.addRow('性别:', self.gender_combo)
        
        # 岗位输入框
        self.position_edit = QLineEdit()
        form_layout.addRow('岗位:', self.position_edit)
        
        # 部门输入框
        self.department_edit = QLineEdit()
        form_layout.addRow('所属部门:', self.department_edit)
        
        # 人员类型选择
        self.type_combo = QComboBox()
        self.type_combo.addItems(['正式员工', '临时工', '实习生', '访客', '其他'])
        form_layout.addRow('人员类型:', self.type_combo)
        
        # 进驻时间选择
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        form_layout.addRow('进驻时间:', self.date_edit)
        
        layout.addLayout(form_layout)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 确定按钮
        confirm_button = QPushButton('确定')
        confirm_button.clicked.connect(self.confirm_register)
        button_layout.addWidget(confirm_button)
        
        # 取消按钮
        cancel_button = QPushButton('取消')
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
    def confirm_register(self):
        """确认注册"""
        # 获取输入的值
        id_text = self.id_edit.text().strip()
        name = self.name_edit.text().strip()
        
        # 验证必填字段
        if not id_text or not name:
            QMessageBox.warning(self, '错误', 'ID和姓名为必填项')
            return
            
        try:
            id = int(id_text)
        except ValueError:
            QMessageBox.warning(self, '错误', 'ID必须是数字')
            return
            
        # 检查ID是否已存在
        if self.face_db.id_exists(id):
            QMessageBox.warning(self, '错误', 'ID已存在，请选择其他ID')
            return
            
        # 收集所有信息
        self.result = {
            'id': id,
            'name': name,
            'gender': self.gender_combo.currentText(),
            'position': self.position_edit.text().strip(),
            'department': self.department_edit.text().strip(),
            'type': self.type_combo.currentText(),
            'entry_date': self.date_edit.date().toString('yyyy-MM-dd')
        }
        
        self.accept() 