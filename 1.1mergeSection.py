#直接用vtkappendfilter，成功
from tecplotDatReader import tecplotDataReader
import vtk
import numpy as np

# 创建reader实例
data_path = r"D:\Project\00Data_E33\E3-22_000609.dat"
reader = tecplotDataReader(data_path)

# 创建append过滤器
append_filter = vtk.vtkAppendFilter()

# 用于存储所有section的压力范围
p_min = float('inf')
p_max = float('-inf')

# 遍历所有sections
print("\n正在读取和合并所有sections...")
for section_idx in range(len(reader.sections)):
    # 读取section数据
    reader.readSection(section_idx)
    i_max, j_max, k_max = reader.sectionDimensions[section_idx]
    #print(f"处理 Section {section_idx}: {i_max}x{j_max}x{k_max}")

    # 创建结构化网格
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(i_max, j_max, k_max)
    
    # 创建点集
    points = vtk.vtkPoints()
    points.Allocate(i_max * j_max * k_max)

    # 获取section数据
    section_data = reader.sectionData[section_idx]

    # 填充点数据
    for i in range(len(section_data[0])):
        points.InsertPoint(i, section_data[0][i], section_data[1][i], section_data[2][i])
    
    grid.SetPoints(points)
    # 创建速度场（4,5,6列）
    velocity = vtk.vtkFloatArray()
    velocity.SetNumberOfComponents(3)
    velocity.SetName("Velocity")
    for i in range(len(section_data[0])):
        velocity.InsertNextTuple3(
                section_data[4][i],  # u
                section_data[5][i],  # v
                section_data[6][i]   # w
        )
    grid.GetPointData().SetVectors(velocity)
    # 添加所有变量场数据
    for var_idx, var_name in enumerate(reader.variables):
        var_array = vtk.vtkFloatArray()
        var_array.SetName(var_name)
        var_data = section_data[var_idx]
        # 添加数据
        for value in var_data:
            var_array.InsertNextValue(value)
            
        grid.GetPointData().AddArray(var_array)
    # 将网格添加到append过滤器
    append_filter.AddInputData(grid)

# 执行合并
append_filter.MergePointsOn()
append_filter.Update()

# 获取合并后的网格
merged_grid = append_filter.GetOutput()
# 创建梯度过滤器
gradient_filter = vtk.vtkGradientFilter()
gradient_filter.SetInputData(merged_grid)
gradient_filter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Velocity")
gradient_filter.Update()
    
# 获取梯度数据
gradients_array = gradient_filter.GetOutput().GetPointData().GetArray("Gradients")
    
# 创建Q值数组
q_criterion = vtk.vtkDoubleArray()
q_criterion.SetNumberOfComponents(1)
q_criterion.SetNumberOfTuples(merged_grid.GetNumberOfPoints())
q_criterion.SetName("Q")
    
# 计算每个点的Q值
for i in range(merged_grid.GetNumberOfPoints()):
    grad = [0.0] * 9
    gradients_array.GetTuple(i, grad)
        
    # Q = 0.5 * (|Ω|^2 - |S|^2)
    q_value = 0.5 * (
        -grad[0] * grad[0] - grad[4] * grad[4] - grad[8] * grad[8]
        - 2 * (grad[2] * grad[6] + grad[5] * grad[7] + grad[1] * grad[3])
        )
    q_criterion.SetValue(i, q_value)
    
    # 将Q值添加到网格数据中
    merged_grid.GetPointData().AddArray(q_criterion)
    merged_grid.GetPointData().SetActiveScalars("Q")

# 直接保存非结构化网格为 VTU 格式
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("merged_grid.vtu")
writer.SetInputData(merged_grid)
writer.Write()


'''
# 创建表面几何过滤器（提取表面三角面片）
geometry_filter = vtk.vtkGeometryFilter()
geometry_filter.SetInputData(merged_grid)
geometry_filter.Update()

# 计算法向量（可选，使模型显示更平滑）
normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(geometry_filter.GetOutputPort())
normals.ComputePointNormalsOn()
normals.Update()

# 创建OBJ写入器
obj_writer = vtk.vtkOBJWriter()
obj_writer.SetFileName("merged_surface.obj")
obj_writer.SetInputConnection(normals.GetOutputPort())
obj_writer.Write()'''

print(f"几何表面已保存为 merged_surface.obj")
print(f"\n合并完成:")
print(f"总点数: {merged_grid.GetNumberOfPoints()}")
print(f"总单元数: {merged_grid.GetNumberOfCells()}")
#print(f"压力范围: [{p_min:.2f}, {p_max:.2f}]")
print(f"数据已保存为 merged_grid.vtu")

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