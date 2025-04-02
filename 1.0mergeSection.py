#用遍历的方法，显式的把i、j、k转换成unstructuredgrid的六面体结构
from tecplotDatReader import tecplotDataReader
import vtk
import numpy as np

# 创建reader实例
data_path = r"D:\Project\00Data_E33\E3-22_000609.dat"
reader = tecplotDataReader(data_path)

# 创建一个unstructured grid
merged_grid = vtk.vtkUnstructuredGrid()
merged_points = vtk.vtkPoints()
merged_pressure = vtk.vtkFloatArray()
merged_pressure.SetName("Pressure")

# 用于存储所有section的压力范围
p_min = float('inf')
p_max = float('-inf')

point_counter = 0  # 用于跟踪点的全局索引

# 遍历所有sections
print("\n正在读取和合并所有sections...")
for section_idx in range(len(reader.sections)):
    # 读取section数据
    reader.readSection(section_idx)
    i_max, j_max, k_max = reader.sectionDimensions[section_idx]
    print(f"处理 Section {section_idx}: {i_max}x{j_max}x{k_max}")

    # 获取坐标和压力数据
    section_data = reader.sectionData[section_idx]
    x_coords = section_data[0]
    y_coords = section_data[1]
    z_coords = section_data[2]
    p_data = section_data[3]

    # 更新压力范围
    p_min = min(p_min, min(p_data))
    p_max = max(p_max, max(p_data))

    # 添加点和压力数据
    for i in range(len(x_coords)):
        merged_points.InsertNextPoint(x_coords[i], y_coords[i], z_coords[i])
        merged_pressure.InsertNextValue(p_data[i])

    # 创建六面体单元
    for k in range(k_max - 1):
        for j in range(j_max - 1):
            for i in range(i_max - 1):
                # 计算六面体的8个顶点索引
                p0 = point_counter + i + j * i_max + k * i_max * j_max
                p1 = p0 + 1
                p2 = p0 + i_max
                p3 = p2 + 1
                p4 = p0 + i_max * j_max
                p5 = p4 + 1
                p6 = p4 + i_max
                p7 = p6 + 1

                # 创建六面体单元
                hexahedron = vtk.vtkHexahedron()
                hexahedron.GetPointIds().SetId(0, p0)
                hexahedron.GetPointIds().SetId(1, p1)
                hexahedron.GetPointIds().SetId(2, p3)
                hexahedron.GetPointIds().SetId(3, p2)
                hexahedron.GetPointIds().SetId(4, p4)
                hexahedron.GetPointIds().SetId(5, p5)
                hexahedron.GetPointIds().SetId(6, p7)
                hexahedron.GetPointIds().SetId(7, p6)

                merged_grid.InsertNextCell(hexahedron.GetCellType(), hexahedron.GetPointIds())

    point_counter += len(x_coords)

# 设置点和数据
merged_grid.SetPoints(merged_points)
merged_grid.GetPointData().AddArray(merged_pressure)
merged_grid.GetPointData().SetActiveScalars("Pressure")

print(f"\n合并完成:")
print(f"总点数: {merged_points.GetNumberOfPoints()}")
print(f"总单元数: {merged_grid.GetNumberOfCells()}")
print(f"压力范围: [{p_min:.2f}, {p_max:.2f}]")

# 创建渲染器和映射器
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.2, 0.4)

mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(merged_grid)
mapper.ScalarVisibilityOn()
mapper.SetScalarRange(p_min, p_max)

# 创建actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 添加颜色条
scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetLookupTable(mapper.GetLookupTable())
scalar_bar.SetTitle("Pressure")
scalar_bar.SetNumberOfLabels(5)

# 设置渲染器
renderer.AddActor(actor)
renderer.AddActor2D(scalar_bar)

# 创建渲染窗口
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1024, 768)

# 创建交互器
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(style)

# 开始渲染
print("\n开始渲染...")
interactor.Initialize()
renderer.ResetCamera()
render_window.Render()
print("渲染窗口已创建，等待交互...")
interactor.Start()