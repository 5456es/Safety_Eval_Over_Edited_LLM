import json
import argparse
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_path', type=str, required=True, help="Path to clean")
    args = parser.parse_args()

    # 检查 clean_path 是否有效
    clean_path = args.clean_path
    if not os.path.isdir(clean_path):
        print(f"Error: {clean_path} is not a valid directory.")
        exit(1)

    # 遍历 clean_path 目录中的所有文件和文件夹
    for item in os.listdir(clean_path):
        item_path = os.path.join(clean_path, item)

        # 保留 "eval" 文件夹和 "log.txt" 文件，删除其他内容
        if item == "eval" and os.path.isdir(item_path):
            print(f"Skipping directory: {item_path}")
            continue
        elif item == "log.txt" and os.path.isfile(item_path):
            print(f"Skipping file: {item_path}")
            continue
        else:
            # 删除文件或文件夹
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")

    print("Cleanup completed.")
