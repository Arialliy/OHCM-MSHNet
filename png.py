# 输入和输出文件路径
input_file = '/Users/regina/Desktop/论文2/数据2/视频/val/val_data88.txt'  # 请替换为你的输入文件路径
output_file = '/Users/regina/Desktop/论文2/数据2/视频/val/val_data88.txt'  # 请替换为你想要保存的输出文件路径

# 打开并读取原文件
with open(input_file, 'r') as f:
    lines = f.readlines()

# 创建一个新的文件，将格式化后的内容写入
with open(output_file, 'w') as f:
    for line in lines:
        # 去除每行的换行符，并将数字格式化为 test_数字
        f.write(f'val_test_{line.strip()}\n')

print("文件已处理并保存。")
