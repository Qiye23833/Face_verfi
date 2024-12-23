import cv2
import numpy as np
import dlib
from PIL import Image
import os
import threading
import queue
import time

class FaceProcessor:
    """
    人脸处理类
    使用dlib进行人脸检测和特征提取
    """
    def __init__(self):
        """初始化人脸处理器"""
        # 加载dlib的人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        
        # 加载人脸关键点检测模型
        model_path = os.path.join(os.path.dirname(__file__), 
                                '../weights/shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(model_path)
        
        # 加载人脸特征提取模型
        rec_model_path = os.path.join(os.path.dirname(__file__),
                                    '../weights/dlib_face_recognition_resnet_model_v1.dat')
        self.face_rec = dlib.face_recognition_model_v1(rec_model_path)
        
        # 用于缓存最近的检测结果
        self.last_detection = None
        self.last_detection_time = 0
        self.detection_interval = 0.5  # 检测间隔（秒）
        
    def detect_face(self, image):
        """
        检测图像中的人脸
        Args:
            image: 输入图像(BGR格式)
        Returns:
            faces: 检测到的人脸区域列表
        """
        current_time = time.time()
        
        # 如果距离上次检测时间不够长，返回缓存的结果
        if (current_time - self.last_detection_time) < self.detection_interval and self.last_detection is not None:
            return self.last_detection
            
        try:
            # 增强图像
            enhanced_image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)  # 提高亮度和对比度
            
            # 转换为灰度图像
            gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            
            # 使用多个尺度进行人脸检测
            faces = self.detector(gray, 1)  # 增加第二个参数，表示上采样次数，提高检测小人脸的能力
            
            # 转换为numpy数组格式的边界框
            face_boxes = []
            for face in faces:
                # 扩大检测框，以包含更多面部区域
                x1 = max(0, face.left() - 10)
                y1 = max(0, face.top() - 20)
                x2 = min(image.shape[1], face.right() + 10)
                y2 = min(image.shape[0], face.bottom() + 10)
                face_boxes.append([x1, y1, x2, y2, 1.0])
            
            face_boxes = np.array(face_boxes)
            
            # 如果没有检测到人脸，尝试使用更宽松的参数
            if len(face_boxes) == 0:
                faces = self.detector(gray, 2)  # 增加上采样次数
                face_boxes = []
                for face in faces:
                    x1 = max(0, face.left() - 20)
                    y1 = max(0, face.top() - 30)
                    x2 = min(image.shape[1], face.right() + 20)
                    y2 = min(image.shape[0], face.bottom() + 20)
                    face_boxes.append([x1, y1, x2, y2, 1.0])
                face_boxes = np.array(face_boxes)
            
            # 更新缓存
            self.last_detection = face_boxes
            self.last_detection_time = current_time
            
            return face_boxes
            
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            return np.array([])
    
    def get_landmarks(self, image, face_box):
        """获取人脸关键点"""
        try:
            x1, y1, x2, y2 = map(int, face_box[:4])
            rect = dlib.rectangle(x1, y1, x2, y2)
            shape = self.predictor(image, rect)
            return shape
        except Exception as e:
            print(f"Landmark detection error: {str(e)}")
            return None
    
    def extract_face_features(self, image, face_box):
        """提取人脸特征"""
        try:
            # 获取人脸关键点
            shape = self.get_landmarks(image, face_box)
            if shape is None:
                return None
            
            # 提取特征
            face_descriptor = self.face_rec.compute_face_descriptor(image, shape)
            
            # 转换为numpy数组并进行L2归一化
            features = np.array(face_descriptor)
            features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None
    
    def align_face(self, image, face_box):
        """对人脸进行对齐和裁剪"""
        try:
            # 获取关键点
            shape = self.get_landmarks(image, face_box)
            if shape is None:
                return None
                
            points = np.array([[p.x, p.y] for p in shape.parts()])
            
            # 获取眼睛关键点的平均位置
            left_eye = points[36:42].mean(axis=0)
            right_eye = points[42:48].mean(axis=0)
            
            # 计算旋转角度
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # 计算缩放比例
            eye_distance = np.linalg.norm(right_eye - left_eye)
            desired_eye_distance = 70
            scale = desired_eye_distance / eye_distance
            
            # 计算旋转中心（确保是整数坐标）
            center_x = int((left_eye[0] + right_eye[0]) / 2)
            center_y = int((left_eye[1] + right_eye[1]) / 2)
            center = (center_x, center_y)
            
            # 计算旋转矩阵
            M = cv2.getRotationMatrix2D(center, angle, scale)
            
            # 进行仿射变换
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            # 裁剪人脸区域
            x1, y1, x2, y2 = map(int, face_box[:4])
            face = aligned[y1:y2, x1:x2]
            
            # 调整大小
            face = cv2.resize(face, (112, 112))
            
            return face
            
        except Exception as e:
            print(f"Face alignment error: {str(e)}")
            return None