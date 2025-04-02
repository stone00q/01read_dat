import vtk

# 创建VTU文件读取器
merged_reader = vtk.vtkXMLUnstructuredGridReader()
merged_reader.SetFileName(r"D:\Project\01read_dat\merged_grid.vtu")  # 替换为实际文件路径
merged_reader.Update()
    
# 获取网格数据
GLOBAL_GRID = merged_reader.GetOutput()
def analyze_Q_distribution():
    """分析Q值分布特征"""
    global GLOBAL_GRID
    
    # 获取Q值数组
    q_array = GLOBAL_GRID.GetPointData().GetArray("Q")
    q_values = [q_array.GetValue(i) for i in range(q_array.GetNumberOfTuples())]
    
    # 基本统计
    q_max = max(q_values)
    q_min = min(q_values)
    q_avg = sum(q_values)/len(q_values)
    q_sorted = sorted(q_values)
    q_median = q_sorted[len(q_sorted)//2]
    
    print(f"Q值统计:")
    print(f"最大值: {q_max:.2e}")
    print(f"最小值: {q_min:.2e}")
    print(f"平均值: {q_avg:.2e}")
    print(f"中位数: {q_median:.2e}")
    
    # 分位数分析
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    for q in quantiles:
        val = q_sorted[int(len(q_sorted)*q)]
        print(f"{q*100:.0f}%分位数: {val:.2e}")

    # 保存不同区间的点云
    thresholds = [
        (0.5*q_max, q_max),    # 前0.5%
        (0.1*q_max, 0.5*q_max),
        (0.01*q_max, 0.1*q_max),
        (q_min, 0.01*q_max)
    ]
    
    for i, (low, high) in enumerate(thresholds):
        threshold_filter = vtk.vtkThresholdPoints()
        threshold_filter.SetInputData(GLOBAL_GRID)
        threshold_filter.ThresholdBetween(low, high)
        threshold_filter.Update()
        
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(f"D:/Project/01read_dat/data_new/Q_segments/Q_region_{i}.vtp")
        writer.SetInputData(threshold_filter.GetOutput())
        writer.Write()

def main():
    analyze_Q_distribution()

if __name__ == "__main__":
    main()