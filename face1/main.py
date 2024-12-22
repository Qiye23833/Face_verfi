import sys
from PyQt5.QtWidgets import QApplication
from face1.ui.main_window import MainWindow
from face1.utils.detector import FaceDetector
from face1.utils.enhancer import ImageEnhancer

def main():
    """
    主程序入口函数
    初始化应用程序，创建检测器、增强器和主窗口，并启动事件循环
    """
    # 创建QT应用程序实例
    app = QApplication(sys.argv)
    
    # 初始化检测器和增强器
    detector = FaceDetector()  # 创建人体检测器实例
    enhancer = ImageEnhancer()  # 创建图像增强器实例
    
    # 创建并显示主窗口
    window = MainWindow(detector, enhancer)
    window.show()
    
    # 启动应用程序的事件循环
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 