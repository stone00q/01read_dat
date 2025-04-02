import vtk
import numpy as np
# from vtk.util import numpy_support as nps
num_particles = 100 # 初始粒子个数
energy_max = 1.0  # 初始能量值
num_steps = 10  # 粒子迭代次数
delta_t = 1 # 积分步长
num_children = 2 # 单次分裂个数
gen_radius = 0.05 # 新分裂出的child粒子与parent粒子距离
generation_max = 5 #最多分裂次数
sphere_radius = 0.005

bounds ={
    'x': (-0.5, 2.5),
    'y': (-2.5, 0.5),
    'z': (-1.5, 1.5)
    }

# 数据递归读取
def extract_unstructured_grid(multi_block_data):
    """递归提取 vtkUnstructuredGrid 块"""
    for i in range(multi_block_data.GetNumberOfBlocks()):
        block = multi_block_data.GetBlock(i)
        if block is None:
            continue
        if isinstance(block, vtk.vtkUnstructuredGrid):
            print(f"Found vtkUnstructuredGrid at block {i}")
            return block
        elif isinstance(block, vtk.vtkMultiBlockDataSet):
            # 递归查找嵌套结构
            result = extract_unstructured_grid(block)
            if result is not None:
                return result
    return None

# 粒子结构
class Particle:
    def __init__(self, position, energy, generation):
        self.position = position  # 粒子位置 (x, y, z)
        #self.velocity = velocity  # 粒子速度 (vx, vy, vz)
        self.energy = energy      # 粒子能量
        self.generation = generation  # 分裂次数

# 粒子Runge Kutta位置更新
# ！遍历所有点，每个点：vtkProbeFilter插值流场速度，然后计算Runge Kutta四阶更新位置
def get_particle_velocity(particle_position, grid, verbose=False):
    """
    使用 VTK 插值流场速度。
    参数:
    particle_position -- 粒子的位置 (x, y, z)
    grid -- VTK 的 unstructured_grid 数据
    返回:
    流场速度向量 (vx, vy, vz)
    """
    # 创建点数据
    points = vtk.vtkPoints()
    points.InsertNextPoint(particle_position)
    # 创建粒子点数据集
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    # 使用 ProbeFilter 对点进行速度插值
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(poly_data)
    probe.SetSourceData(grid)
    probe.Update()
    # 获取插值后的速度
    output = probe.GetOutput()
    velocity_array = output.GetPointData().GetVectors()
    if velocity_array is not None:
        velocity_tuple = velocity_array.GetTuple(0)  # 获取第一个点的速度元组
        velocity_np = np.array(velocity_tuple)  # 将速度元组转换为 numpy 数组
        if verbose:
            print("Interpolated velocity at the particle position:", velocity_np)  # 打印速度值
        return velocity_np  # 返回第一个点的速度作为 numpy 数组
    else:
        print("No velocity data found.")
        return np.array((0.0, 0.0, 0.0)) # 默认返回零向量      
def get_particle_velocity_batch(particles,grid,verbose=False):
    points=vtk.vtkPoints()
    for particle in particles:
        points.InsertNextPoint(particle.position)
    polydata=vtk.vtkPolyData()
    polydata.SetPoints(points)
    probe=vtk.vtkProbeFilter()
    probe.SetInputData(polydata)
    probe.SetSourceData(grid)
    probe.Update()
    output = probe.GetOutput()
    velocity_array = output.GetPointData().GetVectors()
    if velocity_array is not None:
        velocities = [np.array(velocity_array.GetTuple(i)) for i in range(len(particles))]
        if verbose:
            for i in range(10):
                print(f"Particle {i}: Velocity = {velocities[i]}")
    else:
        velocities = [np.array([0.0, 0.0, 0.0])] * len(particles)
    return velocities
def update_particles_position(particles,grid,delta_t,verbose=False):
    velocities=get_particle_velocity_batch(particles,grid)
    for i,particle in enumerate(particles):
        original_position = particle.position.copy()  # 复制原始位置
        # 获取速度，但这个方法好慢？大量重复计算
        # 更新位置
        k1 = velocities[i]
        k2 = get_particle_velocity(particle.position + 0.5 * delta_t * k1,grid)
        k3 = get_particle_velocity(particle.position + 0.5 * delta_t * k2,grid)
        k4 = get_particle_velocity(particle.position + delta_t * k3,grid)
        particle.position += (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # adjust_to_bounds(particle.position)
        if verbose:
            print("Particle:")
            print(f"  Original Position: {original_position}")
            print(f"  Updated Position: {particle.position}")
# 插值所有粒子点的某个指定属性值
def get_field_value_at_particles(particles, grid, property_name,verbose=False):
    points=vtk.vtkPoints()
    for particle in particles:
        points.InsertNextPoint(particle.position)
    polydata=vtk.vtkPolyData()
    polydata.SetPoints(points)
    probe=vtk.vtkProbeFilter()
    probe.SetInputData(polydata)
    probe.SetSourceData(grid)
    probe.Update()
    output = probe.GetOutput()

    point_data=output.GetPointData()
    num_array=point_data.GetNumberOfArrays()
    # for i in range(num_array):
    #     array_name = point_data.GetArrayName(i)
    #     print(f"Array {i}: {array_name}")
    property_array=output.GetPointData().GetArray(property_name)
  
    values = [property_array.GetValue(i) for i in range(property_array.GetNumberOfTuples())]
    if verbose:
        print(f"Interpolated values for {property_name}: {values[:10]}")  # 打印前 10 个值
    return values
# child粒子随机位置，在center周围生成4个child粒子的位置并返回
def generate_children_particle(center, gen_radius, num_children):
    points = []
    for _ in range(num_children):
        # 生成随机的球面坐标
        theta = np.arccos(2 * np.random.rand() - 1)  # 极角 [0, π]
        phi = 2 * np.pi * np.random.rand()           # 方位角 [0, 2π]
        # 转换为直角坐标
        x = center[0] + gen_radius * np.sin(theta) * np.cos(phi)
        y = center[1] + gen_radius * np.sin(theta) * np.sin(phi)
        z = center[2] + gen_radius * np.cos(theta)
        points.append([x, y, z])
    return np.array(points)
# 查看粒子坐标是否越界
def is_within_bounds(position, bounds):
    """
    检查粒子是否在边界内。
    """
    return (bounds['x'][0] <= position[0] <= bounds['x'][1] and
            bounds['y'][0] <= position[1] <= bounds['y'][1] and
            bounds['z'][0] <= position[2] <= bounds['z'][1])
def adjust_to_bounds(position):
    """
    将粒子位置限制在边界范围内。
    """
    # print("有越界粒子")
    position[0] = min(max(position[0], bounds['x'][0]), bounds['x'][1])
    position[1] = min(max(position[1], bounds['y'][0]), bounds['y'][1])
    position[2] = min(max(position[2], bounds['z'][0]), bounds['z'][1])

# 粒子死亡：原地移除能量小于等于0或越界的粒子
def remove_particles(particles):
    particles[:] = [particle for particle in particles if particle.energy > 0 and is_within_bounds(particle.position, bounds)]

# 粒子分裂
def split_particles(particles,g_max,threshold=1.0):
    """
    根据给定的条件对粒子进行分裂。

    参数:
    particles -- Particle 对象列表
    threshold -- 分裂阈值，默认为 1.0
    g_max -- 最大分裂次数，默认为 5
    """

    new_particles = []
    for particle in particles:
        # ？分裂条件为距离源点的距离，之后要换成速度
        distance_from_origin = np.linalg.norm(particle.position)
        if distance_from_origin < threshold and particle.generation < g_max:
            # 进行分裂
            new_positions = generate_children_particle(particle.position,gen_radius,num_children)
            for new_position in new_positions:
                new_particles.append(Particle(
                position=new_position,
                energy=energy_max,
                generation=particle.generation + 1
            ))
    particles.extend(new_particles)
# 粒子按照压力分裂
def split_particles_by_P(particles,grid,g_max):
    new_particles = []
    # 插值获取所有粒子的压力P
    P_array = get_field_value_at_particles(particles, grid, 'P',verbose=True)
    for i, particle in enumerate(particles):
        # 判断压强是否在指定范围内，并且分裂代数未超限
        if abs(P_array[i])<=0.0001 and particle.generation <= g_max:
            # print("p<0.0001, split new particle")
            new_positions = generate_children_particle(particle.position,gen_radius,num_children)
            for new_position in new_positions:
                new_particles.append(Particle(
                position=new_position,
                energy=energy_max,
                generation=particle.generation + 1
            ))
    particles.extend(new_particles)
    new_particles_count = len(new_particles)  # 在分裂步骤中记录
    print(f"此次迭代：New particles generated = {new_particles_count}")


# vtk渲染
def create_particles_renderer_vtk(particles,sphere_radius=0.05):
    """
    创建一个 VTK 渲染器，用于渲染给定的粒子列表为小球。

    参数:
    particles -- Particle 对象列表
    sphere_radius -- 小球半径，默认为 0.05

    返回:
    renderer -- VTK 渲染器对象
    """
    # 创建点数据集
    points = vtk.vtkPoints()
    for particle in particles:
        points.InsertNextPoint(particle.position)

    # 创建顶点（每个点作为一个顶点）
    vertices = vtk.vtkCellArray()
    for i in range(points.GetNumberOfPoints()):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)

    # 创建 PolyData
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetVerts(vertices)

    # 创建球体源
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(sphere_radius)
    sphere_source.Update()

    # 使用 Glyph3D 滤波器将点替换为球体
    glyph3d = vtk.vtkGlyph3D()
    glyph3d.SetSourceConnection(sphere_source.GetOutputPort())
    glyph3d.SetInputData(poly_data)
    glyph3d.ScalingOff()  # 禁用缩放，保持所有球体大小一致
    glyph3d.Update()

    # 创建映射器
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3d.GetOutputPort())

    # 创建演员
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # 创建渲染器并添加演员
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # 设置背景颜色

    return renderer

def save_particles_to_vtp(particles, grid, filename):
    """
    将粒子数据保存为VTP文件
    参数:
    particles: 粒子列表
    grid: 流场网格数据（用于插值获取压力和mut值）
    filename: 保存的文件名
    """
    # 创建点集合
    points = vtk.vtkPoints()
    
    # 创建数据数组
    pressure_array = vtk.vtkFloatArray()
    pressure_array.SetName("Pressure")
    
    mut_array = vtk.vtkFloatArray()
    mut_array.SetName("mut")
    
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
        
        # 添加插值得到的压力和mut值
        pressure = probed_data.GetPointData().GetArray("P").GetValue(i)
        mut = probed_data.GetPointData().GetArray("mut").GetValue(i)
        pressure_array.InsertNextValue(pressure)
        mut_array.InsertNextValue(mut)

    # 创建polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    
    # 添加数据数组
    polydata.GetPointData().AddArray(pressure_array)
    polydata.GetPointData().AddArray(mut_array)
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
if __name__ =="__main__":
    # 1.读取单帧
    # 读取 VTM 文件
    filename = r"D:\Project\00turbine\cal_binary0.12.vtm"
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(filename)
    reader.Update()
    # 获取多块数据
    multi_block_data = reader.GetOutput()
    # 提取 vtkUnstructuredGrid
    unstructured_grid = extract_unstructured_grid(multi_block_data)
    if unstructured_grid is None:
        raise ValueError("No vtkUnstructuredGrid found in the MultiBlock dataset.")
    point_data=unstructured_grid.GetPointData()
    num_array=point_data.GetNumberOfArrays()
    for i in range(num_array):
        array_name = point_data.GetArrayName(i)
        print(f"Array {i}: {array_name}")
    
    # 2.初始化粒子
    # 随机粒子位置
    x_positions = np.random.uniform(bounds['x'][0], bounds['x'][1], num_particles)
    y_positions = np.random.uniform(bounds['y'][0], bounds['y'][1], num_particles)
    z_positions = np.random.uniform(bounds['z'][0], bounds['z'][1], num_particles)
    init_positions = np.stack((x_positions, y_positions, z_positions), axis=1)  # (num_particles, 3)
    # 创建粒子列表
    particles = [
        Particle(position=pos, energy=energy_max,generation=0) 
        for pos in init_positions
    ]

    # 3.粒子迭代的循环
    for step in range(num_steps):
        print(f"第{step+1}次粒子迭代")
        print(f"Step {step+1}:Total particles = {len(particles)}")
        # Step 1: 删除死亡粒子
        remove_particles(particles)
        # Step 2: 能量衰减
        for particle in particles:
            particle.energy -= 0.01  # 简单线性衰减
        # Step 3: 分裂粒子
        #split_particles(particles)
        split_particles_by_P(particles,unstructured_grid,generation_max)
        # Step 4: 粒子位置更新：Runge-Kutta 四阶积分
        update_particles_position(particles,unstructured_grid,delta_t)
        # # Step 5: 渲染粒子（每隔一定步数渲染）
        # render_particles(particles)
        num_out_of_bounds = sum(1 for p in particles if not is_within_bounds(p.position, bounds))
        print(f"Step {step+1}:Particles out of bounds = {num_out_of_bounds}")
        energies = [p.energy for p in particles]
        print(f"Step {step+1}: Avg energy = {np.mean(energies):.4f}, Max energy = {max(energies):.4f}")
        #保存结果为vtp文件
        points = vtk.vtkPoints()
        p_pointdata = vtk.vtkDoubleArray()
        p_pointdata.SetName("P")
        P_array = get_field_value_at_particles(particles, unstructured_grid, 'P')
        for i, particle in enumerate(particles):
            points.InsertNextPoint(particle.position)
            p_pointdata.InsertNextValue(P_array[i])
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(p_pointdata)
        # 写入文件，每个 step 保存一个文件
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(f"particle_step_{step+1}.vtp")  # 动态生成文件名
        writer.SetInputData(polydata)
        writer.Write()
        print(f"Particles saved to particle_step_{step+1}.vtp")

    # 4.结果渲染与保存
    # 将结果保存为vtm，只保存粒子点及其插值压力
    points=vtk.vtkPoints()
    p_pointdata=vtk.vtkDoubleArray()
    p_pointdata.SetName("P")
    P_array = get_field_value_at_particles(particles, unstructured_grid, 'P')
    for i, particle in enumerate(particles):
        points.InsertNextPoint(particle.position)
        p_pointdata.InsertNextValue(P_array[i])
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(p_pointdata)
     # 写入文件
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("particle_binary0.12.vtp")
    writer.SetInputData(polydata)
    writer.Write()
    print("Particles saved to particle_binary0.12")
    # 创建渲染器
    renderer = create_particles_renderer_vtk(particles,sphere_radius)
    # 创建渲染窗口
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    # 创建交互器
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    # 开始交互
    interactor.Initialize()
    render_window.Render()
    interactor.Start()




# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# 使用matplotlib渲染
# def render_particles(particles):
#     """
#     渲染粒子位置
#     """
#     positions = np.array([p.position for p in particles])
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1)
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_zlim(z_min, z_max)
#     plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# def render_particles(particles):
#     """
#     使用 Matplotlib 绘制粒子分布。
#     """
#     positions = np.array([particle.position for particle in particles])
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1)
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_zlim(z_min, z_max)
#     ax.set_title(f"Particle distribution at step {step}")
#     plt.show()


# vtk的渲染管线
# # 转换为 vtkPolyData
# geometry_filter = vtk.vtkGeometryFilter()
# geometry_filter.SetInputData(unstructured_grid)
# geometry_filter.Update()
# poly_data = geometry_filter.GetOutput()
# # 设置点云映射器
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputData(poly_data)
# 创建一个映射器
# mapper = vtk.vtkDataSetMapper()
# mapper.SetInputData(unstructured_grid)
# # 设置点云演员
# actor = vtk.vtkActor()
# actor.SetMapper(mapper)
# # 创建渲染器
# renderer = vtk.vtkRenderer()
# renderer.AddActor(actor)
# renderer.SetBackground(0.1, 0.1, 0.1)  # 设置背景色为深灰
# # 创建渲染窗口
# render_window = vtk.vtkRenderWindow()
# render_window.AddRenderer(renderer)
# render_window.SetSize(800, 600)
# # 创建交互控件
# interactor = vtk.vtkRenderWindowInteractor()
# interactor.SetRenderWindow(render_window)
# # 开始渲染和交互
# render_window.Render()
# interactor.Start()
