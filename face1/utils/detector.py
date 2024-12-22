import torch

class FaceDetector:
    def __init__(self, weights_path='../weights/yolov5s.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        
    def detect(self, frame):
        # 执行检测
        results = self.model(frame)
        return results
    
    def draw_detections(self, results):
        # 在图像上绘制检测结果
        return results.render()[0] 