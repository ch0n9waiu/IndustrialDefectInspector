"""Inspect the structure, annotations, image statistics, and ROI statistics of NEU-DET."""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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

    # Run the complete non-interactive Phase 1 inspection by default.
    inspect_dataset_structure(dataset_path)
    inspect_annotations(dataset_path)
    check_image_annotation_pairs(dataset_path)
    inspect_image_files(dataset_path)
    analyze_class_membership(dataset_path)
    analyze_image_statistics(dataset_path)
    analyze_roi_statistics(dataset_path)

    single_brightness_by_class, single_contrast_by_class = (
        analyze_image_statistics(dataset_path)
    )

    plot_brightness_contrast_distribution(
        single_brightness_by_class,
        single_contrast_by_class,
        "brightness"
    )

    plot_brightness_contrast_distribution(
        single_brightness_by_class,
        single_contrast_by_class,
        "contrast"
    )

    (
        roi_brightness_delta_by_class,
        roi_contrast_delta_by_class
    ) = analyze_roi_statistics(dataset_path)

    plot_roi_delta_distribution(
        roi_brightness_delta_by_class,
        roi_contrast_delta_by_class,
        "brightness"
    )

    plot_roi_delta_distribution(
        roi_brightness_delta_by_class,
        roi_contrast_delta_by_class,
        "contrast"
    )


def inspect_dataset_structure(dataset_path):
    """Inspect dataset folders, image count, and file types."""

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


def inspect_annotations(dataset_path):
    """Inspect XML annotations and bounding boxes."""

    xml_paths = list(dataset_path.rglob("*.xml"))

    defect_class_counter = Counter()

    total_defects = 0
    empty_annotations = 0
    max_objects_per_image = 0

    invalid_box_count = 0
    out_of_bounds_count = 0

    filename_mismatches = []

    min_xmin = float("inf")
    min_ymin = float("inf")
    max_xmax = 0
    max_ymax = 0
    for xml_path in xml_paths:
        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        # Validate the filename stored inside the XML
        xml_filename = xml_root.findtext("filename")

        if xml_filename is None:
            raise ValueError(f"Missing filename in XML: {xml_path}")

        if xml_path.stem != Path(xml_filename).stem:
            filename_mismatches.append(xml_path.name)

        # Validate objects and bounding boxes
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

            min_xmin = min(min_xmin, xmin)
            min_ymin = min(min_ymin, ymin)
            max_xmax = max(max_xmax, xmax)
            max_ymax = max(max_ymax, ymax)

            if xmin > xmax or ymin > ymax:
                invalid_box_count += 1
                print(f"Invalid bbox: {xml_path.name}")

            if xmin < 1 or ymin < 1 or xmax > width or ymax > height:
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

    print("Minimum xmin:", min_xmin)
    print("Minimum ymin:", min_ymin)
    print("Maximum xmax:", max_xmax)
    print("Maximum ymax:", max_ymax)


def check_image_annotation_pairs(dataset_path):
    """Check whether every image has a matching XML annotation."""

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


def inspect_image_files(dataset_path):
    """Check image readability and image/XML size consistency."""

    xml_paths = list(dataset_path.rglob("*.xml"))
    image_paths = list(dataset_path.rglob("*.jpg"))

    xml_map = {
        xml_path.stem: xml_path
        for xml_path in xml_paths
    }

    mismatch_count = 0
    read_error_count = 0

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

    print("\nImage file inspection:")
    print(f"Image read errors: {read_error_count}")
    print(f"Image/XML size mismatches: {mismatch_count}")


def analyze_class_membership(dataset_path):
    """Analyze single-class, multi-class, and class co-occurrence images."""

    xml_paths = list(dataset_path.rglob("*.xml"))

    multi_class_image_count = 0
    class_combination_counter = Counter()
    single_class_image_counter = Counter()

    for xml_path in xml_paths:
        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        xml_objects = xml_root.findall("object")

        image_classes = set()

        for xml_object in xml_objects:
            class_name = xml_object.find("name").text
            image_classes.add(class_name)

        if len(image_classes) > 1:
            multi_class_image_count += 1
            class_combination_counter[frozenset(image_classes)] += 1

        if len(image_classes) == 1:
            class_name = next(iter(image_classes))
            single_class_image_counter[class_name] += 1

    print("\nClass membership analysis:")
    print(f"Multi-class images: {multi_class_image_count}")

    print("Class co-occurrence combinations:")

    for class_comb, count in class_combination_counter.most_common():
        print(f"{sorted(class_comb)}: {count}")

    print("\nSingle-class images:")

    for class_name, count in single_class_image_counter.most_common():
        print(f"{class_name}: {count}")


def analyze_image_statistics(dataset_path):
    """Analyze image brightness and contrast statistics."""

    xml_paths = list(dataset_path.rglob("*.xml"))
    image_paths = list(dataset_path.rglob("*.jpg"))

    xml_map = {
        xml_path.stem: xml_path
        for xml_path in xml_paths
    }

    image_stems = []
    image_means = []
    image_stds = []

    brightness_by_class = defaultdict(list)
    contrast_by_class = defaultdict(list)

    single_brightness_by_class = defaultdict(list)
    single_contrast_by_class = defaultdict(list)
    multi_brightness_by_class = defaultdict(list)
    multi_contrast_by_class = defaultdict(list)

    for image_path in image_paths:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        xml_path = xml_map.get(image_path.stem)

        if xml_path is None:
            continue

        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        brightness = image.mean()
        contrast = image.std()

        image_stems.append(image_path.stem)
        image_means.append(brightness)
        image_stds.append(contrast)

        xml_objects = xml_root.findall("object")

        image_classes = set()

        for xml_object in xml_objects:
            class_name = xml_object.find("name").text
            image_classes.add(class_name)

        # All images containing the current class
        for class_name in image_classes:
            brightness_by_class[class_name].append(brightness)
            contrast_by_class[class_name].append(contrast)

        # Pure single-class images
        if len(image_classes) == 1:
            class_name = next(iter(image_classes))

            single_brightness_by_class[class_name].append(brightness)
            single_contrast_by_class[class_name].append(contrast)
        else:
            for class_name in image_classes:
                multi_brightness_by_class[class_name].append(brightness)
                multi_contrast_by_class[class_name].append(contrast)

    print(
        "\nAll vs single-class vs multi-class "
        "brightness and contrast statistics:"
    )

    for class_name in sorted(single_brightness_by_class):
        multi_brightness_values = multi_brightness_by_class[class_name]
        multi_contrast_values = multi_contrast_by_class[class_name]

        print(f"{class_name}:")
        print(
            f"  all images brightness mean: "
            f"{np.mean(brightness_by_class[class_name]):.2f}"
        )
        print(
            f"  single-class brightness mean: "
            f"{np.mean(single_brightness_by_class[class_name]):.2f}"
        )

        print(
            f"  all images contrast mean: "
            f"{np.mean(contrast_by_class[class_name]):.2f}"
        )
        print(
            f"  single-class contrast mean: "
            f"{np.mean(single_contrast_by_class[class_name]):.2f}"
        )

        if multi_brightness_values:
            print(
                f"  multi-class brightness mean: "
                f"{np.mean(multi_brightness_values):.2f}"
            )
            print(
                f"  multi-class contrast mean: "
                f"{np.mean(multi_contrast_values):.2f}"
            )
        else:
            print("  multi-class brightness mean: N/A")
            print("  multi-class contrast mean: N/A")
        print(f"  multi-class images: {len(multi_brightness_by_class[class_name])}")
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
        f"\nLowest mean-brightness image: "
        f"{image_stems[image_means.index(min(image_means))]} "
        f"({min(image_means):.2f})"
    )

    print(
        f"Highest mean-brightness image: "
        f"{image_stems[image_means.index(max(image_means))]} "
        f"({max(image_means):.2f})"
    )

    print(
        f"Lowest-contrast image: "
        f"{image_stems[image_stds.index(min(image_stds))]} "
        f"({min(image_stds):.2f})"
    )

    print(
        f"Highest-contrast image: "
        f"{image_stems[image_stds.index(max(image_stds))]} "
        f"({max(image_stds):.2f})"
    )
    return single_brightness_by_class, single_contrast_by_class


def plot_brightness_contrast_distribution(
    single_brightness_by_class,
    single_contrast_by_class,
    metric_name
):
    """Plot single-class brightness or contrast distributions."""

    plt.figure(figsize=(10, 6))

    if metric_name == "brightness":
        dict_values = single_brightness_by_class
        plt.ylabel("Mean brightness (gray level)")
        plt.title("Single-class Image Brightness Distribution")
    elif metric_name == "contrast":
        dict_values = single_contrast_by_class
        plt.ylabel("Image contrast (std of gray level)")
        plt.title("Single-class Image Contrast Distribution")
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    defect_classes_list = sorted(dict_values)

    defect_values_list = [
        dict_values[class_name]
        for class_name in defect_classes_list
    ]

    plt.boxplot(
        defect_values_list,
        tick_labels=defect_classes_list,
    )
    plt.xlabel("Defect class")

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def visualize_annotation(dataset_path):
    """Visualize one annotation file and report its ROI statistics."""

    xml_path = next((dataset_path / "ANNOTATIONS").glob("*.xml"))

    image_path = dataset_path / "IMAGES" / f"{xml_path.stem}.jpg"

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    xml_tree = ET.parse(xml_path)
    xml_root = xml_tree.getroot()

    for obj in xml_root.findall("object"):
        class_name = obj.find("name").text

        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # VOC 1-based inclusive -> NumPy 0-based slicing
        roi = gray_image[ymin - 1:ymax, xmin - 1:xmax]

        roi_brightness = roi.mean()
        roi_contrast = roi.std()

        print(
            f"{class_name}: "
            f"brightness={roi_brightness:.2f}, "
            f"contrast={roi_contrast:.2f}, "
            f"shape={roi.shape}"
        )

        print(
            class_name,
            "bbox:",
            xmin, ymin, xmax, ymax,
            "roi shape:",
            roi.shape
        )

        cv2.rectangle(
            image,
            (xmin - 1, ymin - 1),
            (xmax - 1, ymax - 1),
            (0, 255, 0),
            2
        )
    whole_brightness = gray_image.mean()
    whole_contrast = gray_image.std()

    print(
        f"Whole image: "
        f"brightness={whole_brightness:.2f}, "
        f"contrast={whole_contrast:.2f}"
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def analyze_roi_statistics(dataset_path):
    """Analyze per-object ROI brightness/contrast and within-image deltas."""

    xml_paths = (dataset_path / "ANNOTATIONS").glob("*.xml")

    image_map = {
        path.stem: path
        for path in (dataset_path / "IMAGES").glob("*.jpg")
    }

    roi_brightness_by_class = defaultdict(list)
    roi_contrast_by_class = defaultdict(list)

    roi_brightness_delta_by_class = defaultdict(list)
    roi_contrast_delta_by_class = defaultdict(list)

    for xml_path in xml_paths:
        image_path = image_map.get(xml_path.stem)

        if image_path is None:
            raise FileNotFoundError(
                f"No matching image for annotation: {xml_path}"
            )

        gray_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if gray_image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Compute whole-image statistics once per image
        whole_brightness = gray_image.mean()
        whole_contrast = gray_image.std()

        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        for obj in xml_root.findall("object"):
            class_name = obj.find("name").text

            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # VOC 1-based inclusive -> NumPy 0-based slicing
            roi = gray_image[
                ymin - 1:ymax,
                xmin - 1:xmax
            ]

            # Compute each ROI statistic once
            roi_brightness = roi.mean()
            roi_contrast = roi.std()

            roi_brightness_by_class[class_name].append(
                roi_brightness
            )
            roi_contrast_by_class[class_name].append(
                roi_contrast
            )

            brightness_delta = (
                roi_brightness - whole_brightness
            )
            contrast_delta = (
                roi_contrast - whole_contrast
            )

            roi_brightness_delta_by_class[class_name].append(
                brightness_delta
            )
            roi_contrast_delta_by_class[class_name].append(
                contrast_delta
            )

    print("\nROI statistics by class:")

    for class_name in sorted(roi_brightness_by_class):
        brightness_values = roi_brightness_by_class[class_name]
        contrast_values = roi_contrast_by_class[class_name]

        brightness_delta_values = (
            roi_brightness_delta_by_class[class_name]
        )
        contrast_delta_values = (
            roi_contrast_delta_by_class[class_name]
        )

        print(class_name)
        print(f"  objects: {len(brightness_values)}")
        print(
            f"  ROI brightness mean: "
            f"{np.mean(brightness_values):.2f}"
        )
        print(
            f"  ROI contrast mean: "
            f"{np.mean(contrast_values):.2f}"
        )
        print(
            f"  ROI brightness delta mean: "
            f"{np.mean(brightness_delta_values):.2f}"
        )
        print(
            f"  ROI contrast delta mean: "
            f"{np.mean(contrast_delta_values):.2f}"
        )

    total_rois = sum(
        len(values)
        for values in roi_brightness_by_class.values()
    )

    print(f"Total ROIs: {total_rois}")

    return (
        roi_brightness_delta_by_class,
        roi_contrast_delta_by_class
    )


def plot_roi_delta_distribution(
    roi_brightness_delta_by_class,
    roi_contrast_delta_by_class,
    metric_name
):
    """Plot ROI-minus-whole-image brightness or contrast delta distributions."""

    plt.figure(figsize=(10, 6))

    if metric_name == "brightness":
        values_by_class = roi_brightness_delta_by_class
        ylabel = "ROI - whole image brightness"
        title = "ROI Brightness Delta by Defect Class"

    elif metric_name == "contrast":
        values_by_class = roi_contrast_delta_by_class
        ylabel = "ROI - whole image contrast"
        title = "ROI Contrast Delta by Defect Class"

    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    class_names = sorted(values_by_class)

    values_list = [
        values_by_class[class_name]
        for class_name in class_names
    ]

    plt.boxplot(
        values_list,
        tick_labels=class_names
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Defect class")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
