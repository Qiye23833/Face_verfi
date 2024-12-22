import torch

class FaceDetector:
    """
    人体检测器类
    使用YOLOv5模型进行人体检测，可以检测图像或视频中的人体
    """
    
    def __init__(self, weights_path='../weights/yolov5s.pt'):
        """
        初始化检测器
        Args:
            weights_path: YOLOv5模型权重文件的路径，默认使用上级目录中的yolov5s.pt
        """
        # 加载YOLOv5模型
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        # 设置只检测人类（person类别的索引是0）
        self.model.classes = [0]  # 只检测类别索引为0的目标（人）
        
    def detect(self, frame):
        """
        对输入图像进行人体检测
        Args:
            frame: 输入的图像帧（numpy数组格式，BGR颜色空间）
        Returns:
            results: YOLOv5的检测结果对象，包含检测到的边界框、置信度等信息
        """
        # 执行检测
        results = self.model(frame)
        return results
    
    def draw_detections(self, results):
        """
        在图像上绘制检测结果
        Args:
            results: YOLOv5的检测结果对象
        Returns:
            annotated_frame: 绘制了检测框的图像（numpy数组格式，BGR颜色空间）
        """
        # 在图像上绘制检测结果
        return results.render()[0] 