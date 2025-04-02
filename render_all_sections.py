from tecplotDatReader import tecplotDataReader
import vtk
import numpy as np

# 创建reader实例
data_path = r"D:\Project\00Data_E33\E3-22_000609.dat"
reader = tecplotDataReader(data_path)

# 创建渲染器
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.2, 0.4)  # 深蓝色背景

# 用于存储所有section的压力范围
p_min = float('inf')
p_max = float('-inf')

# 遍历所有sections
print("\n正在读取和渲染所有sections...")
for section_idx in range(len(reader.sections)):
    # 读取section数据
    reader.readSection(section_idx)
    i_max, j_max, k_max = reader.sectionDimensions[section_idx]
    #print(f"\nSection {section_idx}:")
    #print(f"网格维度: i_max={i_max}, j_max={j_max}, k_max={k_max}")

    # 创建结构化网格
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(i_max, j_max, k_max)

    # 创建点集
    points = vtk.vtkPoints()
    points.Allocate(i_max * j_max * k_max)

    # 获取坐标数据
    section_data = reader.sectionData[section_idx]
    x_coords = section_data[0]
    y_coords = section_data[1]
    z_coords = section_data[2]

    #print(f"数据点数量: {len(x_coords)}")
    #print(f"坐标范围: X[{min(x_coords):.2f}, {max(x_coords):.2f}]")
    #print(f"         Y[{min(y_coords):.2f}, {max(y_coords):.2f}]")
    #print(f"         Z[{min(z_coords):.2f}, {max(z_coords):.2f}]")

    # 填充点数据
    for i in range(len(x_coords)):
        points.InsertPoint(i, x_coords[i], y_coords[i], z_coords[i])

    grid.SetPoints(points)

    # 添加压力场数据
    if len(section_data) > 3:
        pressure = vtk.vtkFloatArray()
        pressure.SetName("Pressure")
        for p in section_data[3]:
            pressure.InsertNextValue(p)
        grid.GetPointData().AddArray(pressure)
        grid.GetPointData().SetActiveScalars("Pressure")
        
        # 更新全局压力范围
        p_min = min(p_min, min(section_data[3]))
        p_max = max(p_max, max(section_data[3]))
        #print(f"压力范围: [{min(section_data[3]):.2f}, {max(section_data[3]):.2f}]")

    # 为每个section创建mapper和actor
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(grid)
    mapper.ScalarVisibilityOn()
    mapper.SetScalarRange(p_min, p_max)  # 使用全局压力范围

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)

# 创建颜色条
scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetLookupTable(mapper.GetLookupTable())
scalar_bar.SetTitle("Pressure")
scalar_bar.SetNumberOfLabels(5)
renderer.AddActor2D(scalar_bar)

# 创建渲染窗口
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1024, 768)  # 增加窗口大小

# 创建交互器
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(style)

print("\n开始渲染...")
# 开始渲染
interactor.Initialize()
renderer.ResetCamera()
render_window.Render()
print(f"\n总共渲染了 {len(reader.sections)} 个sections")
print(f"全局压力范围: [{p_min:.2f}, {p_max:.2f}]")
print("渲染窗口已创建，等待交互...")
interactor.Start()