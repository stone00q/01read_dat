import vtk

def rotate_and_copy_unstructured_grid(input_grid: vtk.vtkUnstructuredGrid, n: int) -> vtk.vtkMultiBlockDataSet:
    """
    旋转 vtkUnstructuredGrid 并均匀复制 n 份，合成为 vtkMultiBlockDataSet。
    
    :param input_grid: 输入的 vtkUnstructuredGrid 数据。
    :param n: 旋转复制的份数。
    :return: 包含 n 份旋转后数据的 vtkMultiBlockDataSet。
    """
    append_filter = vtk.vtkAppendFilter()
    append_filter.MergePointsOn()              # 开启点合并（去重）
    
    for i in range(n):
        angle = (360.0 / n) * i
        
        # 创建变换
        transform = vtk.vtkTransform()
        transform.RotateX(angle)
        
        # 创建变换过滤器
        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(input_grid)
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        
      
        # 获取变换后的数据
        rotated_grid = vtk.vtkUnstructuredGrid()
        rotated_grid.DeepCopy(transform_filter.GetOutput())
        
        # 添加到 multi block 数据集
        #multi_block.SetBlock(i, rotated_grid)
        #  # 转换为 vtkPolyData
        # geometry_filter = vtk.vtkGeometryFilter()
        # geometry_filter.SetInputData(rotated_grid)
        # geometry_filter.Update()
        # poly_data = vtk.vtkPolyData()
        # poly_data.DeepCopy(geometry_filter.GetOutput())
        
        # # 添加到 multi block 数据集
        # multi_block.SetBlock(i, poly_data)
    # return multi_block
        # 合并所有block成为一个multiblock
        append_filter.AddInputData(rotated_grid)
    append_filter.Update()
    merged_data=append_filter.GetOutput()
    return merged_data
            
    
if __name__ == "__main__":
    # 读入
    merged_reader = vtk.vtkXMLUnstructuredGridReader()
    merged_reader.SetFileName(r"D:\Project\01read_dat\merged_grid.vtu")  # 替换为实际文件路径
    merged_reader.Update()
    input_grid = merged_reader.GetOutput()
    copied = rotate_and_copy_unstructured_grid(input_grid, 38)
     # 保存结果
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName("copied.vtu")
    writer.SetInputData(copied)
    writer.Write()
    # 设置 MultiBlock Mapper
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(copied)
    
    # 创建 Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 创建 Renderer、RenderWindow 和 RenderWindowInteractor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    
    # 添加 Actor 并设置背景颜色
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # 深灰色背景
    
    # 开始渲染循环
    render_window.Render()
    render_window_interactor.Start()

