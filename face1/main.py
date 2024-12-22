import sys

from PyQt5.QtWidgets import QApplication
from face1.ui.main_window import MainWindow
from face1.utils.detector import FaceDetector
from face1.utils.enhancer import ImageEnhancer

def main():
    app = QApplication(sys.argv)
    
    # 初始化检测器和增强器
    detector = FaceDetector()
    enhancer = ImageEnhancer()
    
    # 创建主窗口
    window = MainWindow(detector, enhancer)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 