import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "NEU-DET" / "ANNOTATIONS"
OUTPUT_LABELS_DIR = PROJECT_ROOT / "data" / "NEU-DET-YOLO" / "labels"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


# xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
# print(len(xml_files))
# print(xml_files[:3])


def voc_box_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    x_center = ((xmin - 1) + xmax) / 2
    y_center = ((ymin - 1) + ymax) / 2
    box_width = xmax - xmin + 1
    box_height = ymax - ymin + 1

    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = box_width / image_width
    height_norm = box_height / image_height

    return x_center_norm, y_center_norm, width_norm, height_norm


# print(voc_box_to_yolo(1, 1, 200, 200, 200, 200))
# print(voc_box_to_yolo(51, 101, 100, 150, 200, 200))


def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    obj_list = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        obj_list.append((class_name, xmin, ymin, xmax, ymax))

    return image_width, image_height, obj_list


# xml_path = PROJECT_ROOT / "data" / "NEU-DET" / "ANNOTATIONS" / "crazing_2.xml"
# print(parse_voc_annotation(xml_path))


def annotation_to_yolo_lines(xml_path):
    image_width, image_height, objects = parse_voc_annotation(xml_path)

    yolo_lines = []
    for class_name, xmin, ymin, xmax, ymax in objects:
        x_center, y_center, box_width, box_height = voc_box_to_yolo(
            xmin, ymin, xmax, ymax, image_width, image_height
        )
        class_id = CLASS_NAMES.index(class_name)

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    return yolo_lines


# xml_path = PROJECT_ROOT / "data" / "NEU-DET" / "ANNOTATIONS" / "crazing_2.xml"
# for line in annotation_to_yolo_lines(xml_path):
#     print(line)


def write_yolo_label(xml_path, output_path):
    yolo_lines = annotation_to_yolo_lines(xml_path)

    with open(output_path, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")


# xml_path = PROJECT_ROOT / "data" / "NEU-DET" / "ANNOTATIONS" / "crazing_2.xml"
# output_path = Path("/tmp/crazing_2.txt")
# write_yolo_label(xml_path, output_path)
# print(output_path.read_text())


def convert_all_annotations(annotations_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(annotations_dir.glob("*.xml"))

    for xml_path in xml_files:
        output_path = output_dir / f"{xml_path.stem}.txt"
        write_yolo_label(xml_path, output_path)

    print(f"Converted {len(xml_files)} annotation files.")


# convert_all_annotations(ANNOTATIONS_DIR, OUTPUT_LABELS_DIR)
# print(len(list(OUTPUT_LABELS_DIR.glob("*.txt"))))


def validate_yolo_labels(labels_dir):
    txt_files = sorted(labels_dir.glob("*.txt"))

    total_boxes = 0
    errors = []

    for txt_path in txt_files:
        with open(txt_path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                parts = line.split()

                if len(parts) != 5:
                    errors.append(
                        f"{txt_path.name}:{line_number}: "
                        f"expected 5 fields, got {len(parts)} -> {line.strip()}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])
                except ValueError:
                    errors.append(
                        f"{txt_path.name}:{line_number}: invalid numeric values -> {line.strip()}"
                    )
                    continue

                total_boxes += 1

                if not 0 <= class_id < len(CLASS_NAMES):
                    errors.append(
                        f"{txt_path.name}:{line_number}: invalid class_id={class_id}"
                    )

                if not (
                    0 <= x_center <= 1
                    and 0 <= y_center <= 1
                    and 0 < box_width <= 1
                    and 0 < box_height <= 1
                ):
                    errors.append(
                        f"{txt_path.name}:{line_number}: "
                        f"invalid bounding box -> {line.strip()}"
                    )

    print(f"Label files: {len(txt_files)}")
    print(f"Total boxes: {total_boxes}")
    print(f"Errors: {len(errors)}")

    return errors


# errors = validate_yolo_labels(OUTPUT_LABELS_DIR)
# for error in errors[:10]:
#     print(error)


def yolo_box_to_pixel(
    x_center,
    y_center,
    box_width,
    box_height,
    image_width,
    image_height,
):
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height

    box_width_pixel = box_width * image_width
    box_height_pixel = box_height * image_height

    left = x_center_pixel - box_width_pixel / 2
    right = x_center_pixel + box_width_pixel / 2

    top = y_center_pixel - box_height_pixel / 2
    bottom = y_center_pixel + box_height_pixel / 2

    x1 = round(left)
    y1 = round(top)

    x2 = round(right - 1)
    y2 = round(bottom - 1)

    return x1, y1, x2, y2


# print(
#     yolo_box_to_pixel(
#         0.745,
#         0.7325,
#         0.51,
#         0.275,
#         200,
#         200,
#     )
# )


def visualize_yolo_label(image_path, label_path, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_height, image_width = image.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            parts = line.split()

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            x1, y1, x2, y2 = yolo_box_to_pixel(
                x_center,
                y_center,
                box_width,
                box_height,
                image_width,
                image_height,
            )

            class_name = CLASS_NAMES[class_id]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1,
            )

            cv2.putText(
                image,
                class_name,
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise RuntimeError(f"Failed to save image: {output_path}")

    print(f"Saved visualization: {output_path}")


# image_path = PROJECT_ROOT / "data" / "NEU-DET" / "IMAGES" / "crazing_2.jpg"
# label_path = PROJECT_ROOT / "data" / "NEU-DET-YOLO" / "labels" / "crazing_2.txt"
# output_path = PROJECT_ROOT / "outputs" / "label_validation" / "crazing_2.jpg"
# visualize_yolo_label(image_path, label_path, output_path)


def split_dataset(image_dir, seed=42):
    image_paths = sorted(image_dir.glob("*.jpg"))

    rng = random.Random(seed)
    rng.shuffle(image_paths)

    total = len(image_paths)

    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)

    train_images = image_paths[:train_end]
    val_images = image_paths[train_end:val_end]
    test_images = image_paths[val_end:]

    return train_images, val_images, test_images


IMAGES_DIR = PROJECT_ROOT / "data" / "NEU-DET" / "IMAGES"

# train_images, val_images, test_images = split_dataset(IMAGES_DIR)
#
# print("Train:", len(train_images))
# print("Val:", len(val_images))
# print("Test:", len(test_images))
# print("Total:", len(train_images) + len(val_images) + len(test_images))
#
# train_set = set(train_images)
# val_set = set(val_images)
# test_set = set(test_images)
#
# print(train_set.isdisjoint(val_set))
# print(train_set.isdisjoint(test_set))
# print(val_set.isdisjoint(test_set))


def save_split(image_paths, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for image_path in image_paths:
            f.write(str(image_path) + "\n")


SPLITS_DIR = PROJECT_ROOT / "data" / "NEU-DET-YOLO" / "splits"

# train_images, val_images, test_images = split_dataset(IMAGES_DIR)
# save_split(train_images, SPLITS_DIR / "train.txt")
# save_split(val_images, SPLITS_DIR / "val.txt")
# save_split(test_images, SPLITS_DIR / "test.txt")
