import os
import shutil
import code
base_src_path = '/share1/haoran/czq/gmdex_handover/assets/data/scenes/'
base_dst_path = '/share1/haoran/mobile/Mobile_GenH2R/env/data/mobile_scene'

start_folder = 20000
end_folder = 22149

# 创建目标目录结构
def create_directory_structure(folder_number):
    first_dir = folder_number[:2]
    second_dir = folder_number[2:4]
    third_dir = folder_number[4:6]
    dst_path = os.path.join(base_dst_path, first_dir, second_dir, third_dir)
    os.makedirs(dst_path, exist_ok=True)
    return dst_path

# 复制文件夹
def copy_folder(src_folder, dst_folder):
    src_path = os.path.join(base_src_path, src_folder)
    dst_path = create_directory_structure(src_folder[6:])
    # code.interact(local=dict(globals(), **locals()))
    shutil.copytree(src_path, os.path.join(dst_path, src_folder))

# 遍历文件夹并复制
for folder_number in range(start_folder, end_folder + 1):
    folder_name = f'scene_{str(folder_number).zfill(8)}'
    print(folder_name)
    copy_folder(folder_name, base_dst_path)

print("所有文件夹复制完成。")