import torch


from praatio import tgio

# 指定 TextGrid 文件路径
textgrid_file_path = './data/TIDIGITS-ASE/TEST/TextGrid/TEST-MAN-SW-1381246A.TextGrid'

# 读取 TextGrid 文件
tg = tgio.openTextgrid(textgrid_file_path)

# 打印层级结构
print(tg.tierNameList)

# 获取特定层的标注信息
tier_name = "your_tier_name"
entry_list = tg.tierDict[tier_name].entryList