import vtk

def remove_pointdata_arrays(input_filename, output_filename, arrays_to_remove):
    """
    读取 .vtu 文件，移除指定 PointData 数组，保存结果
    :param input_filename: 输入文件名（如 "copied.vtu"）
    :param output_filename: 输出文件名（如 "cleaned_output.vtu"）
    :param arrays_to_remove: 需要移除的数组名称列表（如 ["x", "y", "z", "u", "v", "w", "ur", "vr", "wr", "rho", "mut", "nut", "mu"]）
    """
    # --------------------- 读取数据 ---------------------
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(input_filename)
    reader.Update()
    unstructured_grid = reader.GetOutput()

    # --------------------- 移除指定 PointData 数组 ---------------------
    point_data = unstructured_grid.GetPointData()
    for array_name in arrays_to_remove:
        if point_data.HasArray(array_name):
            point_data.RemoveArray(array_name)  # 按名称移除数组

    # --------------------- 保存结果 ---------------------
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(unstructured_grid)
    writer.SetDataModeToBinary()  # 使用二进制模式减少文件体积
    writer.Write()

    print(f"处理完成！文件已保存为: {output_filename}")

# 使用示例
if __name__ == "__main__":
    input_file = "copied.vtu"
    output_file = "cleaned_output.vtu"
    arrays_to_remove = ["x", "y", "z", "u", "v", "w", "ur", "vr", "wr", "rho", "mut", "nut", "mu"]
    
    remove_pointdata_arrays(input_file, output_file, arrays_to_remove)