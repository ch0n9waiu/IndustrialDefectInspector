import xml.etree.ElementTree as ET

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

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


# print(voc_box_to_yolo(1, 1, 200, 200, 200,200))
# print(voc_box_to_yolo(51, 101, 100, 150, 200,200))

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


xml_path2 = "/home/chongyu/projects/IndustrialDefectInspector/data/NEU-DET/ANNOTATIONS/crazing_2.xml"


def annotation_to_yolo_lines(xml_path):
    image_width, image_height, objects = parse_voc_annotation(xml_path)


    yolo_lines = []
    for class_name, xmin, ymin, xmax, ymax in objects:
        x_center, y_center, box_width, box_height = voc_box_to_yolo(xmin, ymin, xmax, ymax, image_width,
                                                                    image_height)
        class_id = CLASS_NAMES.index(class_name)

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )
    return yolo_lines
xml_path = "/home/chongyu/projects/IndustrialDefectInspector/data/NEU-DET/ANNOTATIONS/crazing_2.xml"
output_path = "/tmp/crazing_2.txt"


def write_yolo_label(xml_path, output_path):
    yolo_lines = annotation_to_yolo_lines(xml_path)

    with open(output_path, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")

write_yolo_label(xml_path, output_path)