"""Inspect the structure and annotations of the NEU-DET dataset."""
from collections import Counter
from pathlib import Path


def main():
    absolute_path = Path(__file__).resolve()
    print("脚本绝对位置是", absolute_path)
    project_root = absolute_path.parent.parent
    print("根目录是", project_root)
    dataset_path = project_root / "data" / "NEU-DET"
    print("data/NEU-DET 路径是", dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    image_paths = list(dataset_path.rglob("*.jpg"))
    print("Number of JPG images:", len(image_paths))
    suffixes = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            print(item.name)
    for path in dataset_path.rglob("*"):
        if path.is_file():
            suffix = path.suffix
            suffixes.append(suffix)
    suffix_counter = Counter(suffixes)
    for suffix, count in suffix_counter.items():
        print(f"{suffix}: {count}")


if __name__ == "__main__":
    main()
