from tecplotDatReader import tecplotDataReader

# 创建reader实例
data_path = r"D:\Project\00Data_E33\E3-22_000609.dat"
reader = tecplotDataReader(data_path)

# 打印基本信息
#print("文件长度:", reader.fileLen)
#print("段落数量:", len(reader.sections))
#print("段落名称数量:", len(reader.sectionName))
#print("段落起始位置数量:", len(reader.sectionBegin))
#print("段落结束位置数量:", len(reader.sectionEnd))

# 打印所有段的名称
#print("\n数据段信息：")
#reader.printSections()

# 读取所有数据段
print("\n正在读取数据...")
reader.readSection(1)

# 打印数据段信息
#print("\n数据段数据：")
#print("sectionData keys:", list(reader.sectionData.keys()))
#print("sectionLooporder:", reader.sectionLooporder)



# 打印变量名称
#print("\n变量列表：")
#for var in reader.variables:
#    print(var)
