from tecplotDatReader import tecplotDataReader
import vtk
import numpy as np
import os
from joblib import Parallel, delayed
# 粒子参数设置
num_particles = 1500  # 初始粒子数量
energy_max = 1.0     # 初始能量值
num_steps = 100       # 迭代次数
delta_t = 1e-6       # 时间步长或者考虑1.5e-5
num_children = 3     # 分裂时产生的子粒子数
gen_radius = 0.0001    # 子粒子生成半径，？？感觉可能需要和网格尺寸差不多大？
generation_max = 6   # 最大分裂代数
sphere_radius = 0.005  # 渲染球体半径
# 计算单步最大允许位移（比如网格特征尺寸的1/10）
max_displacement = 0.0001  # 可以根据网格尺寸调整
Q_max=0;

class Particle:
    def __init__(self, position, energy, generation):
        self.position = np.array(position)  # 粒子位置
        self.energy = energy                # 粒子能量
        self.generation = generation        # 分裂代数
        #self.lifetime = 0                   # 粒子生命周期，暂时没用到
        #self.nut = 0                        # 存储当前位置的nut值


def initialize_flow_field():
    """初始化流场数据，读入dat并返回一个unstructuredgrid，包括点、nut、压力和速度"""
    print("正在读取流场数据...")
    reader = tecplotDataReader(r"D:\Project\00Data_E33\E3-22_000609.dat")
    
    # 合并所有sections
    append_filter = vtk.vtkAppendFilter()
    append_filter.MergePointsOn()
    
    for section_idx in range(len(reader.sections)):
        reader.readSection(section_idx)
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(*reader.sectionDimensions[section_idx])
        
        points = vtk.vtkPoints()
        section_data = reader.sectionData[section_idx]
        
        # 设置点坐标
        for i in range(len(section_data[0])):
            points.InsertNextPoint(section_data[0][i], section_data[1][i], section_data[2][i])
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

        # 添加nut数据（第14个变量）
        nut_array = vtk.vtkFloatArray()
        nut_array.SetName("nut")
        for nut_value in section_data[14]:  # 使用第14个变量作为nut
            nut_array.InsertNextValue(nut_value)
        grid.GetPointData().AddArray(nut_array)

        # 添加压力数据（第10个变量）
        p_array = vtk.vtkFloatArray()
        p_array.SetName("p")
        for p_value in section_data[10]:  # 使用第14个变量作为nut
            p_array.InsertNextValue(p_value)
        grid.GetPointData().AddArray(p_array)
        
        append_filter.AddInputData(grid)
    
    append_filter.Update()
    return append_filter.GetOutput()

def extract_plane_with_cutter(grid, x_value):
    """
    使用vtkCutter截取x=x_value的平面
    返回平面上的点坐标列表
    """
    print(f"正在使用vtkCutter截取x={x_value}的平面...")
    
    # 创建平面
    plane = vtk.vtkPlane()
    plane.SetOrigin(x_value, 0, 0)
    plane.SetNormal(1, 0, 0)  # x方向法向量
    
    # 创建cutter
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(grid)
    cutter.Update()
    
    # 获取切割结果
    cut_polydata = cutter.GetOutput()
    num_points = cut_polydata.GetNumberOfPoints()
    
    # 收集平面上的点
    plane_points = []
    for i in range(num_points):
        point = np.array(cut_polydata.GetPoint(i))
        plane_points.append(point)
    
    print(f"找到 {len(plane_points)} 个点在x={x_value}平面上")
    return plane_points

def initialize_particles(grid, num_particles, energy_max):
    """从网格点中随机选择位置初始化粒子"""
    print("正在初始化粒子...")
    
    # 获取网格所有点
    points = grid.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    # 随机选择点的索引
    selected_indices = np.random.choice(num_points, size=num_particles, replace=False)
    
    particles = []
    for idx in selected_indices:
        point = np.array(points.GetPoint(idx))
        particles.append(Particle(point, energy_max, 0))
    
    print(f"成功初始化 {len(particles)} 个粒子")
    return particles

def initialize_particles_from_plane(grid, num_particles, energy_max, x_value):
    """
    从x=x_value的平面上随机初始化粒子
    """
    print(f"正在从x={x_value}平面初始化粒子...")
    
    # 获取平面上的点
    plane_points = extract_plane_with_cutter(grid, x_value)
    
    if len(plane_points) == 0:
        raise ValueError(f"在x={x_value}平面上没有找到点")
    
    # 随机选择点
    selected_indices = np.random.choice(len(plane_points), size=num_particles, replace=True)
    
    particles = []
    for idx in selected_indices:
        point = plane_points[idx]
        particles.append(Particle(point, energy_max, 0))
    
    print(f"成功从x={x_value}平面初始化 {len(particles)} 个粒子")
    return particles

def calculate_Q_criterion(grid):
    """计算网格中每个点的Q值"""
    # 创建梯度过滤器
    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(grid)
    gradient_filter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Velocity")
    gradient_filter.Update()
    
    # 获取梯度数据
    gradients_array = gradient_filter.GetOutput().GetPointData().GetArray("Gradients")
    
    # 创建Q值数组
    q_criterion = vtk.vtkDoubleArray()
    q_criterion.SetNumberOfComponents(1)
    q_criterion.SetNumberOfTuples(grid.GetNumberOfPoints())
    q_criterion.SetName("Q")
    
    # 计算每个点的Q值
    for i in range(grid.GetNumberOfPoints()):
        grad = [0.0] * 9
        gradients_array.GetTuple(i, grad)
        
        # Q = 0.5 * (|Ω|^2 - |S|^2)
        q_value = 0.5 * (
            -grad[0] * grad[0] - grad[4] * grad[4] - grad[8] * grad[8]
            - 2 * (grad[2] * grad[6] + grad[5] * grad[7] + grad[1] * grad[3])
        )
        q_criterion.SetValue(i, q_value)
    
    # 将Q值添加到网格数据中
    grid.GetPointData().AddArray(q_criterion)
    grid.GetPointData().SetActiveScalars("Q")

    '''下面都是在打印Q相关信息，比如Q值的bounds'''
    # 计算Q值范围
    global Q_max;
    q_min, Q_max = q_criterion.GetRange()
    print(f"Q值范围: [{q_min:.2e}, {Q_max:.2e}]")
    
    # 统计正Q值的比例
    positive_count = 0
    for i in range(q_criterion.GetNumberOfTuples()):
        if q_criterion.GetValue(i) > 0:
            positive_count += 1
    positive_ratio = positive_count / q_criterion.GetNumberOfTuples()
    print(f"正Q值比例: {positive_ratio:.2%}")
    
    return grid

def get_Q_at_position(position, grid):
    """获取指定位置的Q值"""
    point = vtk.vtkPoints()
    point.InsertNextPoint(position)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(point)
    
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(polydata)
    probe.SetSourceData(grid)
    probe.Update()
    
    return probe.GetOutput().GetPointData().GetArray("Q").GetValue(0)

def get_nut_at_position(position, grid):
    """获取指定位置的nut值"""
    point = vtk.vtkPoints()
    point.InsertNextPoint(position)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(point)
    
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(polydata)
    probe.SetSourceData(grid)
    probe.Update()
    
    return probe.GetOutput().GetPointData().GetArray("nut").GetValue(0)

def get_velocity_at_position(position, grid):
    """获取指定位置的速度"""
    point = vtk.vtkPoints()
    point.InsertNextPoint(position)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(point)
    
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(polydata)
    probe.SetSourceData(grid)
    probe.Update()
    
    velocity = probe.GetOutput().GetPointData().GetVectors("Velocity").GetTuple3(0)
    return np.array(velocity)

def update_particle_position(particle, grid, delta_t):
    """使用RK4更新粒子位置"""
    pos = particle.position
    
    # 获取当前位置的速度
    v1 = get_velocity_at_position(pos, grid)
    velocity_magnitude = np.linalg.norm(v1)
     
    # 计算需要的子步数
    # 预估在当前速度下整个delta_t内的位移
    estimated_displacement = velocity_magnitude * delta_t
    num_substeps = max(1, int(np.ceil(estimated_displacement / max_displacement)))
    sub_dt = delta_t / num_substeps
    
    #if num_substeps > 1:
        #print(f"位置: {pos}, 速度: {velocity_magnitude:.2f}, 子步数: {num_substeps}, 子步长: {sub_dt:.2e}")
    
    # 对每个子步长进行RK4积分
    for substep in range(num_substeps):
        # RK4积分
        k1 = get_velocity_at_position(pos, grid)
        if np.any(np.isnan(k1)):
            #print(f"警告: k1速度计算出现NaN, 位置={pos}")
            return False
            
        k2_pos = pos + 0.5 * sub_dt * k1
        k2 = get_velocity_at_position(k2_pos, grid)
        if np.any(np.isnan(k2)):
            #print(f"警告: k2速度计算出现NaN, 位置={k2_pos}")
            return False
            
        k3_pos = pos + 0.5 * sub_dt * k2
        k3 = get_velocity_at_position(k3_pos, grid)
        if np.any(np.isnan(k3)):
            #print(f"警告: k3速度计算出现NaN, 位置={k3_pos}")
            return False
            
        k4_pos = pos + sub_dt * k3
        k4 = get_velocity_at_position(k4_pos, grid)
        if np.any(np.isnan(k4)):
            #print(f"警告: k4速度计算出现NaN, 位置={k4_pos}")
            return False
        
        # 计算新位置
        new_pos = pos + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 检查新位置是否在网格内
        if not is_particle_alive(Particle(new_pos, particle.energy, particle.generation), grid):
            #print(f"警告: 子步 {substep+1}/{num_substeps} 粒子越界, 位置={new_pos}")
            return False
        
        # 检查位移是否合理
        displacement = np.linalg.norm(new_pos - pos)
        if displacement > max_displacement:
            #print(f"警告: 位移过大 {displacement:.2e}, 位置={new_pos}, 速度={velocity_magnitude:.2f}")
            return False
            
        pos = new_pos
    
    particle.position = pos
    return True
def split_probability_function(q_value):
    """根据Q值计算分裂概率"""
    #q_min, q_max = q_bounds
    #定义了全局变量Q_max来存放Q最大值
    # 定义概率映射函数
    if q_value <= 0:
        return 0.0
    elif q_value < Q_max * 0.1:  # Q值小于最大值的10%
        return 0.1
    elif q_value < Q_max * 0.5:  # Q值小于最大值的50%
        return 0.5
    else:  # Q值大于最大值的50%
        return 1.0
def split_particle_by_nut(particle, grid):
    """基于nut值决定是否分裂粒子"""
    nut_value = get_nut_at_position(particle.position, grid)
    #particle.nut = nut_value    #!!暂时在particle结构里没保存
    
    # 根据论文的方法，设置分裂阈值
    nut_threshold = 1e-10  # 需要根据实际数据调整
    
    if 0 < nut_value < nut_threshold and particle.generation < generation_max:
        new_particles = []
        for _ in range(num_children):
            # 在球面上随机生成新粒子
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            new_pos = particle.position + gen_radius * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            new_particle = Particle(
                position=new_pos,
                energy=energy_max,
                generation=particle.generation + 1
            )
            new_particles.append(new_particle)
        #if new_particles:  # 如果有新粒子生成，打印调试信息
        #    print(f"粒子分裂: nut值={nut_value:.2e}, 位置={particle.position}, 代数={particle.generation}")
            
        return new_particles
    return []

def split_particle_by_Q(particle, grid):
    """基于Q值进行概率分裂粒子"""
    q_value = get_Q_at_position(particle.position, grid)
    #particle.q = q_value
    
    # 设置Q值分裂阈值
    #q_threshold = 0.1  # 需要根据实际数据调整
    # 计算分裂概率
    split_prob = split_probability_function(q_value)
    if np.random.random()< split_prob and particle.generation < generation_max:
        #np.random.random生成一个[0,1]随机数
        new_particles = []
        for _ in range(num_children):
            # 在球面上随机生成新粒子
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            new_pos = particle.position + gen_radius * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            new_particle = Particle(
                position=new_pos,
                energy=energy_max,
                generation=particle.generation + 1
            )
            new_particles.append(new_particle)
            
        #if new_particles:  # 如果有新粒子生成，打印调试信息
            #print(f"粒子分裂: Q值={q_value:.2e}, 位置={particle.position}, 代数={particle.generation}")
            
        return new_particles
    return []

def is_particle_alive(particle, grid):
    """判断粒子是否存活"""
    # 检查能量
    if particle.energy <= 0:
        return False
    
    # 检查是否在网格内
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(grid)
    locator.BuildLocator()
    # 方法1：使用FindCell
    cell_id = locator.FindCell(particle.position)
    return cell_id >= 0  # 如果返回-1表示点不在任何单元内

    # 或者方法2：使用更严格的判断
    """
    tol = 1e-8  # 容差
    closest_point = [0.0, 0.0, 0.0]
    cell_id = vtk.vtkIdType()
    subId = vtk.nutable(0)
    dist2 = vtk.nutable(0.0)
    
    # 获取最近的单元
    found = locator.FindClosestPoint(particle.position, closest_point, cell_id, subId, dist2)
    
    if found == -1:
        return False
        
    # 检查点是否真的在单元内
    cell = grid.GetCell(cell_id)
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0] * cell.GetNumberOfPoints()
    inside = cell.EvaluatePosition(particle.position, closest_point, subId, pcoords, dist2, weights)
    
    return inside == 1
    """

def save_particles_to_vtp(particles, grid, filename):
    """
    将粒子数据保存为VTP文件
    参数:
    particles: 粒子列表
    grid: 流场网格数据（用于插值获取压力和nut值）
    filename: 保存的文件名
    """
    # 创建点集合
    points = vtk.vtkPoints()
    
    # 创建数据数组
    pressure_array = vtk.vtkFloatArray()
    pressure_array.SetName("Pressure")
    
    nut_array = vtk.vtkFloatArray()
    nut_array.SetName("nut")

    q_array = vtk.vtkFloatArray()
    q_array.SetName("Q")
    
    energy_array = vtk.vtkFloatArray()
    energy_array.SetName("Energy")
    
    generation_array = vtk.vtkIntArray()
    generation_array.SetName("Generation")

    # 创建probe filter用于数据插值
    probe = vtk.vtkProbeFilter()
    
    # 收集所有粒子位置
    temp_points = vtk.vtkPoints()
    for particle in particles:
        temp_points.InsertNextPoint(particle.position)
    
    # 创建临时polydata用于插值
    temp_polydata = vtk.vtkPolyData()
    temp_polydata.SetPoints(temp_points)
    
    # 进行插值
    probe.SetInputData(temp_polydata)
    probe.SetSourceData(grid)
    probe.Update()
    
    # 获取插值结果
    probed_data = probe.GetOutput()
    
    # 添加所有粒子数据
    for i, particle in enumerate(particles):
        # 添加位置
        points.InsertNextPoint(particle.position)
        
        # 添加能量和代数
        energy_array.InsertNextValue(particle.energy)
        generation_array.InsertNextValue(particle.generation)
        
        # 添加插值得到的压力和nut值
        pressure = probed_data.GetPointData().GetArray("p").GetValue(i)
        nut = probed_data.GetPointData().GetArray("nut").GetValue(i)
        q_value = probed_data.GetPointData().GetArray("Q").GetValue(i)
        pressure_array.InsertNextValue(pressure)
        nut_array.InsertNextValue(nut)
        q_array.InsertNextValue(q_value)

    # 创建polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    
    # 添加数据数组
    polydata.GetPointData().AddArray(pressure_array)
    polydata.GetPointData().AddArray(nut_array)
    polydata.GetPointData().AddArray(q_array)
    polydata.GetPointData().AddArray(energy_array)
    polydata.GetPointData().AddArray(generation_array)

    # 创建顶点单元，使粒子在ParaView中可见
    vertices = vtk.vtkCellArray()
    for i in range(len(particles)):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)
    polydata.SetVerts(vertices)

    # 写入文件
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"粒子数据已保存到: {filename}")
def save_plane_and_particles(grid, x_value, num_particles, output_dir):
    """
    保存x=x_value的平面和随机生成的粒子
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 截取平面
    plane_points = extract_plane_with_cutter(grid, x_value)
    
    # 2. 随机生成粒子
    selected_indices = np.random.choice(len(plane_points), size=num_particles, replace=True)
    particles = [plane_points[idx] for idx in selected_indices]
    
    # 3. 保存平面
    plane_polydata = vtk.vtkPolyData()
    plane_points_vtk = vtk.vtkPoints()
    for point in plane_points:
        plane_points_vtk.InsertNextPoint(point)
    plane_polydata.SetPoints(plane_points_vtk)
    
    plane_writer = vtk.vtkXMLPolyDataWriter()
    plane_writer.SetFileName(os.path.join(output_dir, f"plane_x={x_value}.vtp"))
    plane_writer.SetInputData(plane_polydata)
    plane_writer.Write()
    
    # 4. 保存粒子
    particles_polydata = vtk.vtkPolyData()
    particles_points_vtk = vtk.vtkPoints()
    for point in particles:
        particles_points_vtk.InsertNextPoint(point)
    particles_polydata.SetPoints(particles_points_vtk)
    
    # 添加顶点
    vertices = vtk.vtkCellArray()
    for i in range(len(particles)):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)
    particles_polydata.SetVerts(vertices)
    
    particles_writer = vtk.vtkXMLPolyDataWriter()
    particles_writer.SetFileName(os.path.join(output_dir, f"particles_x={x_value}.vtp"))
    particles_writer.SetInputData(particles_polydata)
    particles_writer.Write()
    
    print(f"平面和粒子已保存到: {output_dir}")

def update_particles_parallel(particles, flow_field, delta_t):
    """并行更新粒子位置"""
    results = Parallel(n_jobs=-1)(delayed(update_particle_position)(p, flow_field, delta_t) for p in particles)
    #return [p for p, success in zip(particles, results) if success or is_particle_alive(p, flow_field)]
    # 过滤掉不在网格内的粒子
    return [p for p in particles if is_particle_alive(p, flow_field)]

def split_particles_parallel(particles, flow_field):
    """并行分裂粒子"""
    new_particles = Parallel(n_jobs=-1)(delayed(split_particle_by_Q)(p, flow_field) for p in particles)
    return [child for sublist in new_particles for child in sublist]  # 展平列表


def main():
    # 设置粒子保存路径
    particle_output_dir = r"D:\Project\01read_dat\data_new\steps=100-nump=1500"
    # 确保目录存在
    os.makedirs(particle_output_dir, exist_ok=True)
    # 初始化流场

    flow_field = initialize_flow_field()

    # ！！测试：保存x=-0.05的平面和粒子
    #x_value = -0.05
    #num_particles = 1000
    #output_dir = r"D:\Project\01read_dat\data_new\test_plane"
    #save_plane_and_particles(flow_field, x_value, num_particles, output_dir)

    # 计算Q值
    flow_field = calculate_Q_criterion(flow_field)
    # 初始化粒子
    #particles = initialize_particles(flow_field, num_particles, energy_max)
    x_value = -0.05
    particles = initialize_particles_from_plane(flow_field, num_particles, energy_max, x_value)
    # 主循环
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}, Particles: {len(particles)}")
        
        # 1. 更新存活粒子
        new_particles = []
        for particle in particles:
            # 更新位置 
            #if update_particle_position(particle, flow_field, delta_t):
            update_particle_position(particle, flow_field, delta_t)
            # 能量衰减
            #particle.energy -= 0.01  # 可以根据需要调整衰减率
            
            # 如果粒子仍在网格内，考虑分裂
            #if is_particle_alive(particle, flow_field):
                #children = split_particle_by_nut(particle, flow_field)
                #children = split_particle_by_Q(particle, flow_field)
                #new_particles.extend(children)
        
        # 2. 添加新生成的粒子
        particles.extend(new_particles)
        
        # 3.. 移除死亡粒子(考虑，新生成的粒子可能出界)
        particles = [p for p in particles if is_particle_alive(p, flow_field)]
        
        print(f"当前粒子数量: {len(particles)}")
        
        # 4.数据保存
        # 每隔一定步数保存一次，或者保存特定步数
        #if step % 2 == 0 or step == num_steps - 1:  # 每5步保存一次，以及最后一步


        '''并行计算'''
         # 1. 更新存活粒子（并行）
        #particles = update_particles_parallel(particles, flow_field, delta_t)
        
        # 2. 添加新生成的粒子（并行）
        #new_particles = split_particles_parallel(particles, flow_field)
        #new_particles = new_particles[:max_new_particles_per_step]  # 限制数量
        #particles.extend(new_particles)
        # 3.. 移除死亡粒子(考虑，新生成的粒子可能出界)
        #particles = [p for p in particles if is_particle_alive(p, flow_field)]

        filename = os.path.join(particle_output_dir, f"particles_step_{step:04d}.vtp")
        save_particles_to_vtp(particles, flow_field, filename)


if __name__ == "__main__":
    main()