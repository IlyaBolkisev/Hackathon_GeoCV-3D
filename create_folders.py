import os
import shutil
import re


# МЕНЯЙ КАЖДЫЙ РАЗ 3 ВЕЩИ
def move_and_copy_files(source_dir, stl_file_path):
    # Шаблоны для поиска файлов (смотрим на последние три цифры, 001 правое, 002 левое)

    # МЕНЯЙ КАЖДЫЙ РАЗ 3 ВЕЩИ
    pattern_map = re.compile(r'rotateY_pyramid_map(\d{4})(00[12])\.png')
    # МЕНЯЙ КАЖДЫЙ РАЗ 3 ВЕЩИ
    pattern_orig = re.compile(r'rotateY_pyramid_orig(\d{4})(00[12])\.png')

    file_groups = {}

    for filename in os.listdir(source_dir):
        map_match = pattern_map.match(filename)
        orig_match = pattern_orig.match(filename)

        if map_match:
            index = map_match.group(1)
            if index not in file_groups:
                file_groups[index] = {'map': [], 'orig': []}
            file_groups[index]['map'].append(filename)

        elif orig_match:
            index = orig_match.group(1)
            if index not in file_groups:
                file_groups[index] = {'map': [], 'orig': []}
            file_groups[index]['orig'].append(filename)

    folder_counter = 0

    for index, files in file_groups.items():
        map_files = files['map']
        orig_files = files['orig']

        if len(map_files) == 2 and len(orig_files) == 2:

            # МЕНЯЙ КАЖДЫЙ РАЗ 3 ВЕЩИ
            target_dir = os.path.join(source_dir, f"rotateY_pyramid_{folder_counter}")

            os.makedirs(target_dir, exist_ok=True)

            for file in map_files + orig_files:
                src_path = os.path.join(source_dir, file)
                dest_path = os.path.join(target_dir, file)
                shutil.move(src_path, dest_path)

            shutil.copy(stl_file_path, target_dir)

            folder_counter += 1

            print(f"Файлы с индексом {index} перемещены в папку: {target_dir}")
        else:
            print(f"Не хватает файлов для группы с индексом {index}.")



if __name__ == '__main__':
    source_directory = "./dataset_pyramid"
    stl_file = "Pyramid small 5x2.stl"

    move_and_copy_files(source_directory, stl_file)
