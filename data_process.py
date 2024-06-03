import os
import shutil

def move_files_with_extension(source_folder, target_folder, extension, num_files):
    # 创建目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取源文件夹中满足条件的文件列表
    files = [file for file in os.listdir(source_folder) if file.endswith(extension)]
    
    if len(files) < num_files:
        print(f"源文件夹中的文件数量不足 {num_files} 个")
        return

    # 根据文件数量选择要移动的文件
    files_to_move = files[:num_files]

    # 移动文件到目标文件夹
    for file in files_to_move:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        shutil.move(source_path, target_path)
        print(f"移动文件: {file}")

# 示例用法

extension = ".npy"  # 文件后缀名
num_files = 256  # 要移动的文件数量

# move_files_with_extension(source_folder, target_folder, extension, num_files)
for i in range(190):
    data = "{:04d}".format(i)
    source_folder = f"/data/graspnet/rect_labels/scene_{data}/kinect"  # 源文件夹路径
    target_folder = f"/data/graspnet/scenes/scene_{data}/kinect/rect"  # 目标文件夹路径
    print(source_folder)
    print(target_folder)
    # move_files_with_extension(source_folder, target_folder, extension, num_files)
    