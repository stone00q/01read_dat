'''2.2版本粒子数量还是太少了，考虑添加粒子（每次迭代都从初始x=0.05面发射新粒子'''
from scipy.spatial import KDTree
import vtk
import numpy as np
import os
import time
#from joblib import Parallel, delayed
# 全局变量
#GLOBAL_GRID_ORIGIN = None
GLOBAL_GRID = vtk.vtkImageData()
GLOBAL_PROBE_FILTER = vtk.vtkProbeFilter()
GLOBAL_CELL_LOCATOR =None
GLOBAL_PLANE_POINTS =[]
GLOBAL_VALID_ARRAY=None
# 粒子参数设置
num_particles = 1500  # 初始粒子数量
energy_max = 1.0     # 初始能量值
num_steps = 500       # 迭代次数
delta_t = 1.5e-5       # 时间步长或者考虑1.5e-5(尝试了1e-6，移动太慢了)
num_children = 4     # 分裂时产生的子粒子数
gen_radius = 0.0001    # 子粒子生成半径，？？感觉可能需要和网格尺寸差不多大？
generation_max = 5   # 最大分裂代数
sphere_radius = 0.005  # 渲染球体半径
# 计算单步最大允许位移（比如网格特征尺寸的1/10）
max_displacement = 0.0001  # 可以根据网格尺寸调整
Q_MAX=0

class Particle:
    def __init__(self, position, energy, generation):
        self.position = np.array(position)  # 粒子位置
        self.energy = energy                # 粒子能量
        self.generation = generation        # 分裂代数
        #self.lifetime = 0                   # 粒子生命周期，暂时没用到
        #self.nut = 0                        # 存储当前位置的nut值

def source_particles_from_plane(num_particles, energy_max):
    """
    从x=-0.05的平面上随机初始化粒子,返回的是particle列表（particle是粒子坐标）
    """
    print(f"正在从x=-0.05的原始截面（没有插值笛卡尔化）初始化粒子...")
    global GLOBAL_PLANE_POINTS
    # 读取x=-0.05平面上的点
    cutter_reader = vtk.vtkXMLPolyDataReader()
    cutter_reader.SetFileName(r"D:\Project\01read_dat\plane_x=-0.05.vtp")
    cutter_reader.Update()
    # 获取输出数据
    cutter = cutter_reader.GetOutput()
    cutter_num_points = cutter.GetNumberOfPoints()
    # 收集平面上的点
    for i in range(cutter_num_points):
        point = np.array(cutter.GetPoint(i))
        GLOBAL_PLANE_POINTS.append(point)
    
    print(f"找到 {len(GLOBAL_PLANE_POINTS)} 个点在x=-0.05平面上")
    
    
    # 随机选择点
    selected_indices = np.random.choice(len(GLOBAL_PLANE_POINTS), size=num_particles, replace=True)
    
    particles = []
    for idx in selected_indices:
        point = GLOBAL_PLANE_POINTS[idx]
        particles.append(Particle(point, energy_max, 0))
    
    print(f"成功从x=-0.05平面初始化 {len(particles)} 个粒子")
    return particles

def get_particles_Q(particles):
    """获取指定位置的Q值"""
    global GLOBAL_GRID 

    # 创建包含所有粒子位置的临时数据集
    input_points = vtk.vtkPoints()
    for p in particles:
        input_points.InsertNextPoint(p.position)
    input_polydata = vtk.vtkPolyData()
    input_polydata.SetPoints(input_points)
    
    # 使用全局probefilter进行批量插值
    GLOBAL_PROBE_FILTER.SetInputData(input_polydata)
    GLOBAL_PROBE_FILTER.Update()
    probed_data = GLOBAL_PROBE_FILTER.GetOutput()
    
    # 获取所有粒子的Q值数组
    q_array = probed_data.GetPointData().GetArray("Q")
    
    return q_array

def get_velocities(positions):
    """获取所有所需位置的的速度"""
    global GLOBAL_PROBE_FILTER
    points = vtk.vtkPoints()
    #for pos in positions:
        #points.InsertNextPoint(pos)
    for x, y, z in positions:
        points.InsertNextPoint(x, y, z)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    

    GLOBAL_PROBE_FILTER.SetInputData(polydata)
    GLOBAL_PROBE_FILTER.Update()
    
    velocity_data = GLOBAL_PROBE_FILTER.GetOutput().GetPointData().GetVectors("Velocity")
    
    #velocities = []
    #for i in range(len(positions)):
    #    velocities.append(np.array(velocity_data.GetTuple3(i)))
    #return velocities
    return np.array([velocity_data.GetTuple3(i) for i in range(len(positions))])  # 直接返回 NumPy 数组,形状为n*3


def update_particle_position(particles):
    """使用RK4更新粒子位置"""
    global GLOBAL_GRID
    positions = np.array([p.position for p in particles]) 
    
    # 获取当前位置的速度
    k1 = get_velocities(positions)
    velocities_magnitude = np.linalg.norm(k1,axis=1)
    max_velocity_magnitude = np.max(velocities_magnitude)
    print(f"最大速度模值：{max_velocity_magnitude}")
    # 预估在最大速度下整个delta_t内的位移
    estimated_displacement = max_velocity_magnitude*delta_t 
    print(f"预测移动距离：{estimated_displacement}")
    num_substeps = max(1, int(np.ceil(estimated_displacement / max_displacement)))
    sub_dt = delta_t / num_substeps
    print(f"本次迭代的积分步数：{num_substeps}")
    # 对每个子步长进行RK4积分
    for substep in range(num_substeps):
        if substep != 0:
            # 第一个子步长已经计算了 k1，直接用
            k1=get_velocities(positions)
 
        k2_pos = positions + 0.5 * sub_dt * k1
        k2 = get_velocities(k2_pos)
            
        k3_pos = positions + 0.5 * sub_dt * k2
        k3 = get_velocities(k3_pos)
            
        k4_pos = positions + sub_dt * k3
        k4 = get_velocities(k4_pos)
        
        # 计算新位置
        positions += (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    for i, p in enumerate(particles):
        p.position = positions[i]  # 把更新后的位置赋值回 `particles`

# 新增有效性检查函数
def is_position_valid(position):
    """检查位置是否有效（包含边界检查和掩码检查）"""
    # 1. 边界检查
    bounds = GLOBAL_GRID.GetBounds()
    if not (bounds[0] <= position[0] <= bounds[1] and
            bounds[2] <= position[1] <= bounds[3] and
            bounds[4] <= position[2] <= bounds[5]):
        return False
    
    # 2. 掩码有效性检查
    
    if GLOBAL_VALID_ARRAY:
        # 获取最近点的有效性
        closest_point_id = GLOBAL_GRID.FindPoint(position)
        return GLOBAL_VALID_ARRAY.GetValue(closest_point_id) > 0
    return False

def generate_square_tangential_positions(position, velocity):
    """
    计算四个新粒子的位置，使它们分布在源粒子速度方向的切平面内，形成一个以 2 * gen_radius 为边长的正方形。
    """
    velocity_norm = np.linalg.norm(velocity)

    if velocity_norm > 1e-6:
        velocity_dir = velocity / velocity_norm  # 归一化速度方向
    else:
        velocity_dir = np.array([1.0, 0.0, 0.0])  # 速度太小时，默认 x 方向

    # 找到两个与 velocity_dir 垂直的单位切向量
    if abs(velocity_dir[0]) < abs(velocity_dir[1]):  
        tangent1 = np.cross(velocity_dir, np.array([1, 0, 0]))  # 选取 x 轴为参考
    else:
        tangent1 = np.cross(velocity_dir, np.array([0, 1, 0]))  # 选取 y 轴为参考

    tangent1 /= np.linalg.norm(tangent1)  # 归一化
    tangent2 = np.cross(velocity_dir, tangent1)  # 第二个正交向量
    tangent2 /= np.linalg.norm(tangent2)  # 归一化

    # 计算正方形的四个顶点
    offset = gen_radius  # 这里使用 gen_radius，正方形边长 2 * gen_radius
    new_positions = [
        position + offset * (tangent1 + tangent2),  
        position + offset * (tangent1 - tangent2),  
        position + offset * (-tangent1 + tangent2),  
        position + offset * (-tangent1 - tangent2)  
    ]

    return new_positions
def split_probability_function(q_value):
    """根据Q值计算分裂概率,这里要不要修改，改成根据每次迭代的粒子中，q的最大值，即看局部最值进行概率映射"""
    #q_min, q_max = q_bounds
    #定义了全局变量Q_max来存放Q最大值
    # 定义概率映射函数
    if q_value <= 0 :
        return 0.0
    '''elif q_value < Q_MAX * 0.1:  # Q值小于最大值的10%
        return 0.1
    elif q_value < Q_MAX * 0.5:  # Q值小于最大值的50%
        return 0.5
    else:  # Q值大于最大值的50%
        return 1.0'''
    q_scaled = (q_value - 6.27e+08) / (4.00e+09 - 6.27e+08)  # 归一化
    return np.clip(1 / (1 + np.exp(-5 * (q_scaled - 0.5))), 0, 1)  # 使用 sigmoid

def split_particles_by_Q(particles, q_array):
    """基于Q值进行概率分裂粒子,返回的是新增的粒子且一定在流场范围内"""
    new_particles = []

     # 先构造 KDTree，方便查询局部粒子密度
    positions = np.array([p.position for p in particles])
    kdtree = KDTree(positions)

    #对于每个点需要知道速度，然后子粒生成与原始粒子速度相切
    positions = np.array([p.position for p in particles]) 
    velocities = get_velocities(positions)
    # 逐个点判断是否分裂
    for i,particle in enumerate(particles):
        q_value = q_array.GetValue(i)
        # 计算分裂概率
        split_prob = split_probability_function(q_value)

         # 查询周围粒子数（搜索半径为 2 * gen_radius）
        num_neighbors = len(kdtree.query_ball_point(particle.position, 2 * gen_radius))
         # 根据局部密度调整分裂概率
        if num_neighbors > 10:  # 若周围有超过 10 个粒子，则降低分裂概率
            split_prob *= 0.2  # 只保留 20% 的原始概率
        elif num_neighbors > 20:  # 若粒子非常密集，则完全不分裂
            continue
        if np.random.random()< split_prob and particle.generation < generation_max:
            #np.random.random生成一个[0,1]随机数
            new_positions=generate_square_tangential_positions(positions[i], velocities[i])
            for new_pos in new_positions:
                 # 新增位置有效性检查
                if is_position_valid(new_pos):
                    new_particle = Particle(
                        position=new_pos,
                        energy=energy_max,
                        generation=particle.generation + 1
                    )
                    new_particles.append(new_particle)
            
    return new_particles


def save_particles_to_vtp(particles,filename):
    """
    将粒子数据保存为VTP文件
    参数:
    particles: 粒子列表
    filename: 保存的文件名
    """
    global GLOBAL_PROBE_FILTER

    input_points = vtk.vtkPoints()
    for p in particles:
        input_points.InsertNextPoint(p.position)

    input_polydata = vtk.vtkPolyData()
    input_polydata.SetPoints(input_points)

     # 设置probefilter输入并执行插值
    GLOBAL_PROBE_FILTER.SetInputData(input_polydata)
    GLOBAL_PROBE_FILTER.Update()  # 每次修改input后需要Update
    probed_data = GLOBAL_PROBE_FILTER.GetOutput()

     # 添加粒子特有属性（直接从内存数据添加）
    energy_array = vtk.vtkFloatArray()
    energy_array.SetName("Energy")
    generation_array = vtk.vtkIntArray()
    generation_array.SetName("Generation")
    
    for p in particles:
        energy_array.InsertNextValue(p.energy)
        generation_array.InsertNextValue(p.generation)
    # 将自定义属性添加到probefilter输出数据
    probed_data.GetPointData().AddArray(energy_array)
    probed_data.GetPointData().AddArray(generation_array)

    # 创建顶点单元（直接使用probefilter输出的点结构）使粒子在ParaView中可见
    vertices = vtk.vtkCellArray()
    for i in range(probed_data.GetNumberOfPoints()):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)
    probed_data.SetVerts(vertices)
    
    # 写入文件
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(probed_data)
    writer.Write()
    
    print(f"粒子数据已保存到: {filename}")


def main():
    # 全局变量
    global GLOBAL_GRID, GLOBAL_PROBE_FILTER,GLOBAL_CELL_LOCATOR,Q_MAX,GLOBAL_VALID_ARRAY

    # 设置粒子保存路径
    particle_output_dir = r"D:\Project\01read_dat\data_new\steps=500-nump=1500-Cartesian2"
    # 确保目录存在
    os.makedirs(particle_output_dir, exist_ok=True)
    '''读入流场数据'''
    start_time = time.time()
    
    # 创建vti阅读器
    merged_reader = vtk.vtkXMLImageDataReader()
    merged_reader.SetFileName(r"D:\Project\01read_dat\cartesian_grid_auto.vti")
    merged_reader.Update()
    # 获取网格数据
    GLOBAL_GRID = merged_reader.GetOutput()
    GLOBAL_VALID_ARRAY = GLOBAL_GRID.GetPointData().GetArray("vtkValidPointMask")
    GLOBAL_GRID.GetPointData().RemoveArray("vtkValidPointMask")

    q_prop = GLOBAL_GRID.GetPointData().GetArray("Q")
    Q_MAX = q_prop.GetRange()[1]  # GetRange() 返回 (min, max)
    print(f"Q_Max={Q_MAX}")
    GLOBAL_PROBE_FILTER.SetSourceData(GLOBAL_GRID)
    # 创建适合ImageData的定位器
    GLOBAL_CELL_LOCATOR = vtk.vtkStaticCellLocator() if isinstance(GLOBAL_GRID, vtk.vtkImageData) else vtk.vtkCellLocator()
    GLOBAL_CELL_LOCATOR.SetDataSet(GLOBAL_GRID)
    GLOBAL_CELL_LOCATOR.BuildLocator()
    particles=[]

    # 主循环
    for step in range(num_steps):
        new_source = source_particles_from_plane(num_particles, energy_max)
        particles.extend(new_source)#每次迭代，从相同源点发射新的粒子
        print(f"Step {step + 1}/{num_steps}, Particles: {len(particles)}")
        step_start_time = time.time()
        
        # 1. 更新粒子位置
        t1 = time.time()
        update_particle_position(particles)
        t2 = time.time()
        update_time = t2 - t1

        # 2. 移除死亡粒子(考虑，新生成的粒子可能出界)
        t1 = time.time()
        particles = [p for p in particles if p.energy > 0 and is_position_valid(p.position)]
        t2 = time.time()
        remove_time = t2 - t1

        # 3. 粒子分裂
        t1 = time.time()
        q_array = get_particles_Q(particles)
        new_particles = split_particles_by_Q(particles, q_array)
        particles.extend(new_particles)
        t2 = time.time()
        split_time = t2 - t1
        
        print(f"当前粒子数量: {len(particles)}")
        
        # 4.数据保存
        t1 = time.time()
        filename = os.path.join(particle_output_dir, f"particles_step_{step:04d}.vtp")
        save_particles_to_vtp(particles, filename)
        t2 = time.time()
        save_time = t2 - t1
        # 统计本次循环总时间
        step_time = time.time() - step_start_time
         # 输出每个步骤的耗时
        print(f"迭代 Step {step + 1}: 更新位置 {update_time:.6f} s, 移除粒子 {remove_time:.6f} s, "
              f"粒子分裂 {split_time:.6f} s, 数据保存 {save_time:.6f} s, 总耗时 {step_time:.6f} s")
    total_time = time.time() - start_time
    print(f"总运行时间: {total_time:.6f} s")

if __name__ == "__main__":
    main()