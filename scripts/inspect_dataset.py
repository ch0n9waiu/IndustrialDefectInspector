"""
1. 找到当前脚本的位置
2. 向上定位到项目根目录
3. 拼出 data/NEU-DET 路径
4. 判断目录是否存在
5. 递归统计 .jpg 图片数量
6. 打印结果
"""
from pathlib import Path

def main():
    absolute_path = Path(__file__).resolve()
    print("脚本绝对位置是",absolute_path)
    project_root = absolute_path.parent.parent
    print("根目录是",project_root)
    dataset_path =  project_root/"data"/"NEU-DET"
    print("data/NEU-DET 路径是",dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    image_paths = list(dataset_path.rglob("*.jpg"))
    print(len(image_paths))

if __name__ == "__main__":
    main()
