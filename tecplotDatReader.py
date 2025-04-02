import numpy as np
import re

class tecplotDataReader(object):
    def __init__(self,datafile):
        self.clear()
        self.datafile=datafile
        # self.fileLen is using the throwaway returnvalue of the _getVariables method
        self.fileLen= self._getVariables() 
        self._getSections()
    
    def __iter__(self):
        return self
    def __next__(self):
        if self.index == len(self.sectionLooporder):
            raise StopIteration
        self.index = self.index + 1
        return self.sectionLooporder[self.index-1]
    def resetIter(self):
        self.index = 0 #index索引指的是zone的索引   
#    
#    def __repr__(self):
#        return self.sections
#    
#    def __str__(self):
#        pass

    def clear(self):
        self.datafile=""
        self.fileLen=0
        self.sections=[]
        self.sectionName=[]
        self.sectionBegin=[]
        self.sectionEnd=[]
        self.sectionDimensions=[]
        self.sectionData={}
        self.sectionLooporder=[]
        self.variables=[]
        self.index=0
        
    def _getVariables(self):
        with open(self.datafile) as f:
            i=0
            start_reading = False
            for line in f:
                i += 1
                line_stripped = line.strip()  # 去除空白符
                # 检查 'VARIABLES' 标记
                if line_stripped.upper().startswith('VARIABLES'):
                    start_reading = True
                    continue  # 跳过 'VARIABLES' 这一行本身
                
                # 如果已经开始读取变量
                if start_reading:
                    # 遇到下一个关键字，例如 'ZONE'，表示变量部分结束
                    if line_stripped.upper().startswith('ZONE') or line_stripped == "":
                        start_reading = False
                        continue  # 退出循环
                    # 提取变量名
                    self.variables.append(line_stripped)    
        
        # 打印调试信息
        #print(f"提取的变量: {self.variables}") 
        #print(f"文件长度: {i}")
        return i
    
    def _getSections(self):
        with open(self.datafile) as f:
            i = 0
            for line in f:
                i += 1
                # 添加调试信息
                #if i < 20:  # 只打印前20行用于调试
                #    print(f"Line {i}: {line.strip()}")
                
                # 改进ZONE识别
                if "ZONE" in line.upper():
                    # print(f"Found ZONE at line {i}: {line.strip()}")
                    # 尝试不同的匹配模式
                    zone_matches = re.findall(r'["\'](.+?)["\']', line)
                    if zone_matches:
                        self.sectionName.append(zone_matches[0])
                    continue

                if "I =" in line.upper():
                    # print(f"Found IJK at line {i}: {line.strip()}")
                    ijk_matches = re.findall(r'I\s*=\s*(\d+),\s*J\s*=\s*(\d+),\s*K\s*=\s*(\d+)', line.upper())
                    self.sectionBegin.append(i+1)
                    if ijk_matches:
                        I, J, K = map(int, ijk_matches[0])
                        total_nodes = I * J * K
                        self.sectionDimensions.append((I, J, K))
                        self.sectionEnd.append(i + total_nodes)
            
            # 打印找到的节点信息
            # print(f"\n找到的段落数: {len(self.sectionName)}")
            # print(f"段落名称: {self.sectionName}")
            # print(f"起始位置: {self.sectionBegin}")
            # print(f"结束位置: {self.sectionEnd}")
            
            self.sections = list(range(len(self.sectionName))) # 为每个zone创建一个索引列表，索引从0开始
            self.sectionLooporder = list(self.sections)
        return i
    
    def readSection(self, section):
        # 打印调试信息
        #print(f"读取section {section}:")
        #print(f"变量数量: {len(self.variables)}")
        #print(f"开始行: {self.sectionBegin[section]}")
        #print(f"结束行: {self.sectionEnd[section]}")
        
        # 先读取几行数据来检查格式
        #print("\n检查数据格式:")
        #with open(self.datafile) as f:
        #    lines = f.readlines()
        #    start_line = int(self.sectionBegin[section])-1
        #    for i in range(5):  # 打印前5行数据
        #    line = lines[start_line + i].strip()
        #    values = line.split()
        #    print(f"行 {start_line + i}: 列数={len(values)}, 数据={values}")

        data = np.genfromtxt(
            self.datafile,
            delimiter = None,
            skip_header=(int(self.sectionBegin[section])-1),
            skip_footer=(self.fileLen-int(self.sectionEnd[section])),# skip_footer=(self.fileLen-int(self.sectionEnd[section])1772172)
            filling_values=0,
            invalid_raise=False,
            unpack=True,
            # usecols=range(len(self.variables))
        )
        
        # 检查读取的数据
        #print("\n读取的数据形状:")
        #if isinstance(data, np.ndarray):
            #print(f"数据形状: {data.shape}")
            #print(f"数据类型: {data.dtype}")
            #print(f"前几行数据:\n{data[:, :5] if len(data.shape) > 1 else data[:5]}")
        #else:
            #print(f"数据类型: {type(data)}")
                
        self.sectionData[self.sections[section]] = data

    def readAllSections(self):
        for i in range(len(self.sections)):
            self.readSection(i)
    
    def moveSectionToEnd(self,section):
        temp=self.sectionLooporder.pop(section)
        self.sectionLooporder.append(temp)
    
    def printSections(self):#打印所有section的name
        tempindex=self.index
        self.index=0
        for section in self:
            print(str(section) + self.sectionName[section])
        self.index=tempindex
        
    def reorderedData(self,index):
        return self.sectionData[self.sectionLooporder[index]]

