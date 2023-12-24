import subprocess
import tkinter as tk  
import sys
import open3d as o3d
import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget ,QPushButton, QMainWindow, QFileDialog ,QHBoxLayout, QLineEdit, QMessageBox, QComboBox
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QAction
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class SubWindow(QWidget):
    # 在类级别定义信号
    trainFinished = pyqtSignal(str)  # 定义一个新的信号，可以传递一个字符串参数
    def __init__(self):
        super().__init__()
        self.setGeometry(100,100,500,200) 
        self.setWindowTitle("曲面重建参数选择")
        self.centerOnScreen()
        self.setWindowIcon(QIcon('img/sub.jpg'))

        # 添加用于选择参数的控件，比如文本框或下拉菜单
        self.gpuLineEdit = QLineEdit()
        self.modelComboBox = QComboBox()
        self.dataNameLineEdit = QLineEdit()

        self.gpuLineEdit.setPlaceholderText("GPU编号")
        self.modelComboBox.addItems(["validate_mesh", "train"])
        self.dataNameLineEdit.setPlaceholderText("数据名称")

        self.gpu_value = '0'
        self.model_value = 'validate_mesh'
        self.data_name_value = ''
        self.command = ''

        # 添加训练按钮
        self.trainButton = QPushButton('训练')

        # 调用响应事件
        self.trainButton.clicked.connect(self.onTrainButtonClick)

        # 将控件添加到布局中
        layout = QVBoxLayout()
        layout.addWidget(self.gpuLineEdit)
        layout.addWidget(self.modelComboBox)
        layout.addWidget(self.dataNameLineEdit)
        layout.addWidget(self.trainButton)
        self.setLayout(layout)

    def onTrainButtonClick(self):
        self.gpu_value = self.gpuLineEdit.text()
        self.model_value = self.modelComboBox.currentText()
        self.data_name_value = self.dataNameLineEdit.text()
        self.command = f"conda run -n base python run.py --gpu {self.gpu_value} --conf confs/npull.conf --mode {self.model_value} --dataname {self.data_name_value} --dir {self.data_name_value}"
        if self.gpu_value and self.model_value and self.data_name_value:
            self.surfaceReconstruction(self.command)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("不能输入空值!")
            msg.setWindowTitle("Error")
            msg.exec()
    
        # 标准化窗口
    def centerOnScreen(self):
        screen = QApplication.primaryScreen()
        resolution = screen.availableGeometry()
        self.setGeometry((resolution.width() / 2) - (self.frameSize().width() / 2),
                         (resolution.height() / 2) - (self.frameSize().height() / 2),
                         self.frameSize().width(), self.frameSize().height())

        # 曲面重建
    def surfaceReconstruction(self,command):
        print(command)
        # 命令
        command2 = "python test.py"  # 定义另一个命令字符串

        # 创建子进程 将标准输出和错误输出重定向到管道中
        process = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # 使用subprocess.Popen创建一个新的进程，将标准输出和错误输出重定向到管道中

        output_window = tk.Tk()  # 创建一个新的顶层窗口
        output_window.title("训练监控")  # 设置窗口标题为"Subprocess Output"
        output_window.geometry("400x300") # 设置窗口大小
        output_window.geometry("+100+100") # 设置窗口位置

        output_label = tk.Label(output_window, text="Waiting for output...\n")  # 创建一个标签，显示文本"Waiting for output..."
        output_label.pack()  # 将标签放置到窗口中


        def update_output():  # 定义一个函数update_output
            output = process.stdout.readline()  # 读取子进程的标准输出
            print(output)

            if output:  # 如果读取到输出
                output_label.config(text=output_label.cget("text") + output.decode("utf-8"))  # 将输出解码为utf-8格式并添加到标签的文本内容
                output_label.after(100, update_output)  # 在100毫秒后再次调用update_output函数
            else:  # 如果没有读取到输出
                output_label.config(text="训练完毕！")
                self.trainFinished.emit(self.data_name_value)  # 发出信号，传递self.data_name_value作为参数


        output_label.after(100, update_output)  # 在100毫秒后调用update_output函数

        output_window.mainloop()  # 进入tkinter的主事件循环，显示窗口并响应用户操作

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ParameterWindow()
    window.show()
    ret = app.exec()
    del window
    sys.exit(ret)