import cv2
import numpy as np

class ImageEnhancer:
    @staticmethod
    def histogram_equalization(frame):
        # 直方图均衡化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def adjust_brightness_contrast(frame, brightness=0, contrast=1):
        # 亮度和对比度调整
        return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    
    @staticmethod
    def denoise(frame):
        # 降噪处理
        return cv2.fastNlMeansDenoisingColored(frame)
    
    def enhance(self, frame):
        # 组合多个增强方法
        enhanced = self.histogram_equalization(frame)
        enhanced = self.adjust_brightness_contrast(enhanced, brightness=10, contrast=1.2)
        return enhanced 