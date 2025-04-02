import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy

# 读取 VTU 文件
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("cleaned_output.vtu")
reader.Update()

# 获取数据
GLOBAL_GRID = reader.GetOutput()

# 获取网格的边界框信息
bounds = GLOBAL_GRID.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
print(f"数据边界: {bounds}")

# 定义平面
plane = vtk.vtkPlane()
plane.SetOrigin(-0.05, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)  # 选取中心高度
plane.SetNormal(1, 0, 0)  # x方向法向量

# 创建 Cutter
cutter = vtk.vtkCutter()
cutter.SetCutFunction(plane)
cutter.SetInputData(GLOBAL_GRID)
cutter.Update()

# 获取切割结果
cut_polydata = cutter.GetOutput()
num_points = cut_polydata.GetNumberOfPoints()

if num_points == 0:
    print("警告：没有切割到任何点，请检查输入数据！")
else:
    print(f"切割后点数量: {num_points}")

# 提取所有切割到的点
vtk_points = cut_polydata.GetPoints()
numpy_points = vtk_to_numpy(vtk_points.GetData())

# 随机选择 5000 个点
num_selected = min(3000, len(numpy_points))
selected_indices = np.random.choice(len(numpy_points), num_selected, replace=False)
selected_points = numpy_points[selected_indices]

# 创建 VTK 点数据结构
output_points = vtk.vtkPoints()
for p in selected_points:
    output_points.InsertNextPoint(p)

# 创建 PolyData
output_poly_data = vtk.vtkPolyData()
output_poly_data.SetPoints(output_points)

# 写入 VTP 文件
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("source_3k.vtp")
writer.SetInputData(output_poly_data)
writer.Write()

print(f"最终选取点数量: {num_selected}")
print("VTP 文件已保存：output_points.vtp")
