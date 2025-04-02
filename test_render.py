from tecplotDatReader import tecplotDataReader
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkRenderingCore import (vtkRenderer, vtkRenderWindow, 
                                        vtkRenderWindowInteractor, vtkActor, 
                                        vtkDataSetMapper)
from vtkmodules.vtkCommonCore import vtkPoints, vtkFloatArray
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor

import numpy as np
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
# 创建reader实例
data_path = r"D:\Project\00Data_E33\E3-22_000609.dat"
reader = tecplotDataReader(data_path)

# 读取所有数据段
print("\n正在读取数据...")
reader.readSection(1)
# 获取网格维度
i_max, j_max, k_max = reader.sectionDimensions[1]
print(f"网格维度: i_max={i_max}, j_max={j_max}, k_max={k_max}")

# 创建结构化网格
grid = vtkStructuredGrid()
grid.SetDimensions(i_max, j_max, k_max)
# 创建点集
points = vtkPoints()
points.Allocate(i_max * j_max * k_max)
# 获取坐标数据（前三列）
section_data = reader.sectionData[1]  # 获取section 1的数据
x_coords = section_data[0]  # 第一列是x坐标
y_coords = section_data[1]  # 第二列是y坐标
z_coords = section_data[2]  # 第三列是z坐标

# 填充点数据
for i in range(len(x_coords)):
    points.InsertPoint(i, x_coords[i], y_coords[i], z_coords[i])

grid.SetPoints(points)

# 添加标量场数据（例如压力场）
if len(section_data) > 3:  # 确保有额外的数据场
    pressure = vtkFloatArray()
    pressure.SetName("Pressure")  # 设置数据场名称
    for p in section_data[3]:  # 假设第4列是压力数据
        pressure.InsertNextValue(p)
    grid.GetPointData().AddArray(pressure)
    grid.GetPointData().SetActiveScalars("Pressure")

# 设置渲染器和映射器
mapper = vtkDataSetMapper()
mapper.SetInputData(grid)
mapper.ScalarVisibilityOn()
mapper.SetScalarRange(pressure.GetRange())  # 设置标量范围

# 创建颜色条
scalar_bar = vtkScalarBarActor()
scalar_bar.SetLookupTable(mapper.GetLookupTable())
scalar_bar.SetTitle("p")
scalar_bar.SetNumberOfLabels(5)

# 创建actor
actor = vtkActor()
actor.SetMapper(mapper)

# 设置渲染器
renderer = vtkRenderer()
renderer.AddActor(actor)
renderer.AddActor2D(scalar_bar)
renderer.SetBackground(0.1, 0.2, 0.4)  # 深蓝色背景

# 创建渲染窗口
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)

# 创建交互器
interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 开始渲染
interactor.Initialize()
render_window.Render()
interactor.Start()