import sys
from PyQt5.QtWidgets import QApplication
from face1.ui.main_window import MainWindow

def main():
    """
    主程序入口函数
    初始化应用程序，创建主窗口，并启动事件循环
    """
    # 创建QT应用程序实例
    app = QApplication(sys.argv)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 启动应用程序的事件循环
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 