import cv2
import numpy as np

class ImageEnhancer:
    """
    图像增强处理类
    提供多种图像增强方法，用于改善输入图像的质量
    """
    
    @staticmethod
    def histogram_equalization(frame):
        """
        直方图均衡化处理
        通过调整图像直方图来增强图像对比度
        
        Args:
            frame: 输入图像（numpy数组格式，BGR颜色空间）
        Returns:
            enhanced_frame: 经过直方图均衡化的图像
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def adjust_brightness_contrast(frame, brightness=0, contrast=1):
        """
        调整图像的亮度和对比度
        
        Args:
            frame: 输入图像
            brightness: 亮度调整值，正值增加亮度，负值降低亮度
            contrast: 对比度调整值，大于1增加对比度，小于1降低对比度
        Returns:
            adjusted_frame: 调整后的图像
        """
        return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    
    @staticmethod
    def denoise(frame):
        """
        图像降噪处理
        使用非局部均值去噪算法去除图像噪声
        
        Args:
            frame: 输入图像
        Returns:
            denoised_frame: 降噪后的图像
        """
        return cv2.fastNlMeansDenoisingColored(frame)
    
    def enhance(self, frame):
        """
        组合多个图像增强方法
        
        Args:
            frame: 输入图像
        Returns:
            enhanced_frame: 经过多重增强处理的图像
        """
        # 应用直方图均衡化
        enhanced = self.histogram_equalization(frame)
        # 调整亮度和对比度
        enhanced = self.adjust_brightness_contrast(enhanced, brightness=10, contrast=1.2)
        return enhanced