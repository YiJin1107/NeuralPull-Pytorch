import sys
import os
import open3d as o3d
import numpy as np
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget ,QPushButton, QMainWindow, QFileDialog ,QHBoxLayout
from PyQt6 import QtCore
from PyQt6.QtGui import QIcon, QAction
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from subWindow import SubWindow


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 子窗口
        self.parameterWindow = SubWindow()
        self.setWindowIcon(QIcon('img/main.png'))
        self.parameterWindow.trainFinished.connect(self.showSurfaceAfter)  # 连接信号到showSurfaceAfter函数
        # 退出行为
        self.exitAct = QAction(QIcon('exit.png'), '&Exit', self) 
        self.exitAct.setShortcut('Ctrl+Q') 
        self.exitAct.setStatusTip('Exit application') 
        self.exitAct.triggered.connect(QApplication.instance().quit) 

        # 点云选择
        self.cloudAct = QAction('&点云选择', self) 
        self.cloudAct.triggered.connect(self.showCloud) 

        # 曲面选择
        self.surfaceAct = QAction('&曲面选择', self) 
        self.surfaceAct.triggered.connect(self.showSurface) 

        # 曲面重建
        self.trainAct = QAction('&曲面重建', self) 
        self.trainAct.triggered.connect(self.openParameterWindow)  # 连接到打开参数窗口的函数


        # 场景一
        self.frame1 = QVTKRenderWindowInteractor(self) # 用于在Qt窗口显示VTK渲染窗口
        self.ren1 = vtk.vtkRenderer() # 用于渲染3D场景
        self.frame1.GetRenderWindow().AddRenderer(self.ren1)
        self.iren1 = self.frame1.GetRenderWindow().GetInteractor() # 获取frame的交互器
        # 场景二
        self.frame2 = QVTKRenderWindowInteractor(self) # 用于在Qt窗口显示VTK渲染窗口
        self.ren2 = vtk.vtkRenderer() # 用于渲染3D场景
        self.frame2.GetRenderWindow().AddRenderer(self.ren2)
        self.iren2 = self.frame2.GetRenderWindow().GetInteractor() # 获取frame的交互器

        # 定时器 同步窗口
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.syncCameras)

        # 设置UI界面
        self.initUI()


    def startTimer(self):
        self.timer.start(100)
        
    def closeEvent(self, event):
        '''
        窗口关闭时进行垃圾回收
        '''
        # 停止计时器
        self.timer.stop()  
        # 清除frame
        self.frame1.close()
        self.frame2.close()
        super().closeEvent(event)

    def initUI(self):
        '''
        设计UI
        '''
        # 设置状态栏
        self.statusBar()
        
        # 菜单栏
        menubar = self.menuBar() 

        fileMenu = menubar.addMenu('&File') 
        fileMenu.addAction(self.exitAct) 
        fileMenu.addAction(self.cloudAct) 
        fileMenu.addAction(self.surfaceAct) 
        fileMenu.addAction(self.trainAct) 

        # 创建一个水平布局
        hbox = QHBoxLayout()

        # 将frame1和frame2添加到布局中
        hbox.addWidget(self.frame1)
        hbox.addWidget(self.frame2)

        # 创建一个QWidget，将布局设置到这个QWidget
        widget = QWidget()
        widget.setLayout(hbox)

        # 将QWidget设置为主窗口的中心部件
        self.setCentralWidget(widget)


        self.setGeometry(100,100,1200,800) 
        self.centerOnScreen()

        self.setWindowTitle('点云曲面重建') 
        self.show()

    # 标准化窗口
    def centerOnScreen(self):
        screen = QApplication.primaryScreen()
        resolution = screen.availableGeometry()
        self.setGeometry((resolution.width() / 2) - (self.frameSize().width() / 2),
                         (resolution.height() / 2) - (self.frameSize().height() / 2),
                         self.frameSize().width(), self.frameSize().height())
    
    # 显示点云数据
    def showCloud(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "请选择点云：", '.', "cloud Files(*pcd *ply)")
        if fileName != '':
            # 点云操作
            pcd = o3d.io.read_point_cloud(fileName)  # 读取点云
            np_points = np.asarray(pcd.points)  # 读取点云坐标
            colors = np.asarray(pcd.colors)  # 获取点云的颜色信息
            center = np.mean(np_points, axis=0)  # 计算点云中心位置
            centered_points = np_points - center  # 将点云坐标减去中心位置的偏移量
            max_coord = np.max(np.abs(centered_points)) # 计算点云最大值
            centered_points /= max_coord # 归一化处理
            # 创建点云数据
            points = vtk.vtkPoints()
            vertices = vtk.vtkCellArray()
            colorData = vtk.vtkUnsignedCharArray()  # 创建颜色数组
            colorData.SetNumberOfComponents(3)  # 设置颜色组件数量
            colorData.SetName("Colors")  # 设置颜色数组名称

            for i in range(len(centered_points)):
                id = points.InsertNextPoint(centered_points[i])
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(id)
                if colors.size > 0:
                    colorData.InsertNextTuple3(colors[i][0]*255, colors[i][1]*255, colors[i][2]*255)  # 将颜色信息添加到颜色数组中

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetVerts(vertices)
            polydata.GetPointData().SetScalars(colorData)  # 将颜色数组添加到polydata中

            # 创建映射器和actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            self.ren1.AddActor(actor)
            self.ren1.ResetCamera()
            self.frame1.GetRenderWindow().Render()

    # 显示曲面数据
    def showSurface(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "请选择曲面文件：", '.', "Surface Files(*.ply)")
        if fileName != '':
            # 读取PLY文件
            reader = vtk.vtkPLYReader()
            reader.SetFileName(fileName)
            reader.Update()

            # 创建映射器和actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            self.ren2.AddActor(actor)
            self.ren2.ResetCamera()
            self.frame2.GetRenderWindow().Render()

            # 归一化处理
            bounds = actor.GetBounds()
            max_bound = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            actor.SetScale(1/max_bound, 1/max_bound, 1/max_bound)
            actor.SetPosition(-bounds[0]/max_bound, -bounds[2]/max_bound, -bounds[4]/max_bound)
            self.ren2.ResetCamera()
            self.frame2.GetRenderWindow().Render()
    
    # 显示曲面数据
    def showSurfaceAfter(self,fileName):
        print(fileName)
        if fileName != '':
            path = f'{os.getcwd()}/outs/{fileName}/outputs/'
            for _,_,files in os.walk(path):
                for f in files:
                    file_path = f"{path}/{f}"
                    # 读取PLY文件
                    reader = vtk.vtkPLYReader()
                    reader.SetFileName(file_path)
                    reader.Update()

                    # 创建映射器和actor
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(reader.GetOutputPort())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)

                    self.ren2.AddActor(actor)
                    self.ren2.ResetCamera()
                    self.frame2.GetRenderWindow().Render()
                    # 归一化处理
                    bounds = actor.GetBounds()
                    max_bound = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
                    actor.SetScale(1/max_bound, 1/max_bound, 1/max_bound)
                    actor.SetPosition(-bounds[0]/max_bound, -bounds[2]/max_bound, -bounds[4]/max_bound)
                    self.ren2.ResetCamera()
                    self.frame2.GetRenderWindow().Render()
                    # self.startTimer()
                    return


    # 显示子窗口用于选择参数
    def openParameterWindow(self):
        self.parameterWindow.show()
        print('打开子窗口')   

    def syncCameras(self):
        # 获取场景一摄像机参数
        camera1 = self.ren1.GetActiveCamera()
        position1 = camera1.GetPosition()
        focal_point1 = camera1.GetFocalPoint()
        view_up1 = camera1.GetViewUp()

        # 将参数赋值给场景二相机
        camera2 = self.ren2.GetActiveCamera()
        position2 = camera2.GetPosition()
        focal_point2 = camera2.GetFocalPoint()
        view_up2 = camera2.GetViewUp()
        
        position1 = np.array(position1)
        position2 = np.array(position2)
        position = (position1 + position2) // 2


        focal_point1 = np.array(focal_point1)
        focal_point2 = np.array(focal_point2)
        focal_point = (focal_point1 + focal_point2) / 2

        view_up1 = np.array(view_up1)
        view_up2 = np.array(view_up2)
        view_up = (view_up1 + view_up2) / 2

        if np.all(position1 == position2) and np.all(view_up1 == view_up2):
            return
        camera1.SetPosition(position)
        camera2.SetPosition(position)
        camera1.SetFocalPoint(focal_point)
        camera2.SetFocalPoint(focal_point)
        camera1.SetViewUp(view_up)
        camera2.SetViewUp(view_up)


        # 渲染场景二
        self.frame1.GetRenderWindow().Render()
        self.frame2.GetRenderWindow().Render()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()

    window.show()
    ret = app.exec()
    del window
    sys.exit(ret)