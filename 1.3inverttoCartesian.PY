import vtk
import numpy as np
def auto_determine_dimensions(grid):
    """
    根据原始网格自动确定合适的插值分辨率
    
    参数:
    grid: vtkUnstructuredGrid对象
    
    返回:
    dimensions: [nx, ny, nz] 建议的网格分辨率
    """
    # 获取网格边界
    bounds = grid.GetBounds()
    
    # 计算每个维度的范围
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    
    # 获取原始网格的点数
    n_points = grid.GetNumberOfPoints()
    
    # 估算每个维度的大致点数（假设点密度均匀）
    total_volume = x_range * y_range * z_range
    points_density = n_points / total_volume
    
    # 根据点密度计算建议的分辨率
    suggested_dx = (total_volume / n_points) ** (1/3)
    
    # 计算每个维度的网格数
    nx = max(2, int(x_range / suggested_dx))
    ny = max(2, int(y_range / suggested_dx))
    nz = max(2, int(z_range / suggested_dx))
    
    return [nx, ny, nz]
def interpolate_grid_auto(vtu_file, accuracy_factor=1.0):
    """
    将VTU非结构化网格插值到笛卡尔网格
    
    参数:
    vtu_file: VTU文件路径
    bounds: 目标网格边界 [xmin,xmax, ymin,ymax, zmin,zmax]
    dimensions: 目标网格分辨率 [nx,ny,nz]
    """
    # 1. 读取VTU文件
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file)
    reader.Update()
    grid = reader.GetOutput()

    # 2. 获取网格边界
    bounds = grid.GetBounds()
    
    
    # 3. 自动确定分辨率
    base_dimensions = auto_determine_dimensions(grid)
    dimensions = [max(2, int(d * accuracy_factor)) for d in base_dimensions]

    print(f"原始网格点数: {grid.GetNumberOfPoints()}")
    print(f"建议的网格分辨率: {dimensions}")
    print(f"网格范围: X[{bounds[0]:.3f}, {bounds[1]:.3f}]")
    print(f"          Y[{bounds[2]:.3f}, {bounds[3]:.3f}]")
    print(f"          Z[{bounds[4]:.3f}, {bounds[5]:.3f}]")
    
    # 4. 创建目标笛卡尔网格
    cartesian_grid = vtk.vtkImageData()
    cartesian_grid.SetDimensions(dimensions)
    cartesian_grid.SetOrigin(bounds[0], bounds[2], bounds[4])
    
    spacing = [
        (bounds[1]-bounds[0])/(dimensions[0]-1),
        (bounds[3]-bounds[2])/(dimensions[1]-1),
        (bounds[5]-bounds[4])/(dimensions[2]-1)
    ]
    cartesian_grid.SetSpacing(spacing)
    
    # 5. 使用probe filter进行插值
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(cartesian_grid)
    probe_filter.SetSourceData(grid)
    probe_filter.Update()
    
    # 6. 提取有效区域
    '''threshold = vtk.vtkThreshold()
    threshold.SetInputData(probe_filter.GetOutput())
    threshold.SetInputArrayToProcess(0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "ValidPoints")
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    threshold.SetLowerThreshold(1)
    threshold.SetUpperThreshold(1)
    threshold.Update()'''
    
    # 7. 保存结果
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName("cartesian_grid_auto.vti")
    writer.SetInputData(probe_filter.GetOutput())
    writer.Write()
    
    print(f"插值后的点数: {probe_filter.GetOutput().GetNumberOfPoints()}")
    # print(f"有效点数: {threshold.GetOutput().GetNumberOfPoints()}")
    
    return probe_filter.GetOutput()


def main():
    
    vtu_file = "./merged_grid.vtu"
    # accuracy_factor > 1 提高精度
    # accuracy_factor < 1 降低精度
    result = interpolate_grid_auto(vtu_file, accuracy_factor=1.0)

if __name__ == "__main__":
    main()