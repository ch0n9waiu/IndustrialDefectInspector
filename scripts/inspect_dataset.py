"""Inspect the structure, annotations, and image statistics of the NEU-DET dataset."""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    dataset_path = project_root / "data" / "NEU-DET"

    print(f"Script path: {script_path}")
    print(f"Project root: {project_root}")
    print(f"Dataset path: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    image_paths = list(dataset_path.rglob("*.jpg"))
    print(f"Number of JPG images: {len(image_paths)}")

    for item in dataset_path.iterdir():
        if item.is_dir():
            print(item.name)

    suffixes = []

    for path in dataset_path.rglob("*"):
        if path.is_file():
            suffixes.append(path.suffix)

    suffix_counter = Counter(suffixes)

    for suffix, count in suffix_counter.items():
        print(f"{suffix}: {count}")

    inspect_annotations(dataset_path)
    check_image_annotation_pairs(dataset_path)
    inspect_images(dataset_path)
    single_class_image_counter(dataset_path)

def inspect_annotations(dataset_path):
    xml_paths = list(dataset_path.rglob("*.xml"))

    defect_class_counter = Counter()

    total_defects = 0
    empty_annotations = 0
    max_objects_per_image = 0

    invalid_box_count = 0
    out_of_bounds_count = 0

    filename_mismatches = []

    for xml_path in xml_paths:
        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        # 检查 XML 内部 filename
        xml_filename = xml_root.find("filename").text

        if xml_path.stem != Path(xml_filename).stem:
            filename_mismatches.append(xml_path.name)

        # object 和 bbox 检查
        xml_objects = xml_root.findall("object")

        xml_size = xml_root.find("size")
        width = int(xml_size.find("width").text)
        height = int(xml_size.find("height").text)

        for xml_object in xml_objects:
            class_name = xml_object.find("name").text
            defect_class_counter[class_name] += 1

            bbox = xml_object.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            if xmin >= xmax or ymin >= ymax:
                invalid_box_count += 1
                print(f"Invalid bbox: {xml_path.name}")

            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                out_of_bounds_count += 1
                print(f"Out-of-bounds bbox: {xml_path.name}")

        total_defects += len(xml_objects)

        if len(xml_objects) == 0:
            empty_annotations += 1

        max_objects_per_image = max(
            max_objects_per_image,
            len(xml_objects)
        )

    print("\nDefect objects by class:")

    for class_name in sorted(defect_class_counter):
        print(f"{class_name}: {defect_class_counter[class_name]}")

    print(f"Number of XML annotations: {len(xml_paths)}")
    print(f"Total defect objects: {total_defects}")
    print(f"Empty annotations: {empty_annotations}")
    print(f"Max objects in one image: {max_objects_per_image}")
    print(f"Invalid bounding boxes: {invalid_box_count}")
    print(f"Out-of-bounds boxes: {out_of_bounds_count}")
    print(f"Filename mismatches: {len(filename_mismatches)}")

    if filename_mismatches:
        for filename in filename_mismatches:
            print(f"Filename mismatch: {filename}")


def check_image_annotation_pairs(dataset_path):
    xml_paths = list(dataset_path.rglob("*.xml"))
    image_paths = list(dataset_path.rglob("*.jpg"))

    image_name_set = set()
    xml_name_set = set()

    for image_path in image_paths:
        image_name_set.add(image_path.stem)

    for xml_path in xml_paths:
        xml_name_set.add(xml_path.stem)

    images_without_annotations = image_name_set - xml_name_set
    annotations_without_images = xml_name_set - image_name_set

    print("\nImage/XML pairing:")
    print(f"Images without XML: {len(images_without_annotations)}")
    print(f"XML without images: {len(annotations_without_images)}")

    if images_without_annotations:
        for image_stem in sorted(images_without_annotations):
            print(image_stem)

    if annotations_without_images:
        for xml_stem in sorted(annotations_without_images):
            print(xml_stem)


def inspect_images(dataset_path):
    xml_paths = list(dataset_path.rglob("*.xml"))
    image_paths = list(dataset_path.rglob("*.jpg"))

    xml_map = {
        xml_path.stem: xml_path
        for xml_path in xml_paths
    }

    mismatch_count = 0
    read_error_count = 0

    image_stems = []
    image_means = []
    image_stds = []

    brightness_by_class = defaultdict(list)
    contrast_by_class = defaultdict(list)
    multi_class_image_count = 0
    class_combination_counter = Counter()

    single_brightness_by_class = defaultdict(list)
    all_brightness_by_class = defaultdict(list)
    single_contrast_by_class = defaultdict(list)


    for image_path in image_paths:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f"Image read error: {image_path}")
            read_error_count += 1
            continue

        image_height, image_width = image.shape

        xml_path = xml_map.get(image_path.stem)

        if xml_path is None:
            continue

        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        # Image/XML 尺寸检查
        xml_size = xml_root.find("size")

        width = int(xml_size.find("width").text)
        height = int(xml_size.find("height").text)

        if width != image_width or height != image_height:
            mismatch_count += 1
            print(
                f"Size mismatch: {xml_path.name} "
                f"XML=({width}, {height}) "
                f"Image=({image_width}, {image_height})"
            )

        # 当前图片的图像统计
        brightness = image.mean()
        contrast = image.std()

        image_stems.append(image_path.stem)
        image_means.append(brightness)
        image_stds.append(contrast)

        # 当前图片包含的类别
        xml_objects = xml_root.findall("object")

        image_classes = set()

        for xml_object in xml_objects:
            class_name = xml_object.find("name").text
            image_classes.add(class_name)

        # 按 image-level 统计类别亮度和对比度
        for class_name in image_classes:
            brightness_by_class[class_name].append(brightness)
            contrast_by_class[class_name].append(contrast)

        if len(image_classes) > 1:
            multi_class_image_count += 1
            class_combination_counter[frozenset(image_classes)] += 1

        if len(image_classes) == 1:
            class_name = next(iter(image_classes))

            single_brightness_by_class[class_name].append(brightness)
            single_contrast_by_class[class_name].append(contrast)

    for class_name in sorted(single_brightness_by_class):
        print(f"{class_name}:")
        print(
            f"  all images brightness mean: "
            f"{np.mean(brightness_by_class[class_name]):.2f}"
        )
        print(
            f"  single-class brightness mean: "
            f"{np.mean(single_brightness_by_class[class_name]):.2f}"
        )

    print(f"同一张图包含多个类别的图片共: {multi_class_image_count}")
    print(f"组合分别是: ")
    for class_comb, count in class_combination_counter.most_common():
        print(f"{sorted(class_comb)}: {count}")

    print("\nImage statistics by class:")

    for class_name in sorted(brightness_by_class):
        brightness_values = brightness_by_class[class_name]
        contrast_values = contrast_by_class[class_name]

        brightness_mean = np.mean(brightness_values)
        brightness_std = np.std(brightness_values)
        contrast_mean = np.mean(contrast_values)

        print(class_name)
        print(f"  images: {len(brightness_values)}")
        print(f"  brightness mean: {brightness_mean:.2f}")
        print(f"  brightness std: {brightness_std:.2f}")
        print(f"  contrast mean: {contrast_mean:.2f}")

    print(
        f"\n最低平均亮度图片: "
        f"{image_stems[image_means.index(min(image_means))]} "
        f"({min(image_means):.2f})"
    )

    print(
        f"最高平均亮度图片: "
        f"{image_stems[image_means.index(max(image_means))]} "
        f"({max(image_means):.2f})"
    )

    print(
        f"最低对比度图片: "
        f"{image_stems[image_stds.index(min(image_stds))]} "
        f"({min(image_stds):.2f})"
    )

    print(
        f"最高对比度图片: "
        f"{image_stems[image_stds.index(max(image_stds))]} "
        f"({max(image_stds):.2f})"
    )

    print(f"Image read errors: {read_error_count}")
    print(f"Image/XML size mismatches: {mismatch_count}")

def single_class_image_counter(dataset_path):
    xml_paths = list(dataset_path.rglob("*.xml"))
    number_of_each_class = Counter()
    for xml_path in xml_paths:

        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()
        xml_objects = xml_root.findall("object")

        image_classes = set()

        for xml_object in xml_objects:
            class_name = xml_object.find("name").text
            image_classes.add(class_name)

        if len(image_classes) == 1:
            class_name = next(iter(image_classes))
            number_of_each_class[class_name] += 1

    for class_name,count in number_of_each_class.most_common():
        print(f"{class_name}: {count}")



if __name__ == "__main__":
    main()
