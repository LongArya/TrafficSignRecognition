import os
import json
from collections import defaultdict
from typing import Any
from pycocotools.coco import COCO
from pprint import pprint
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime

# image_path, bbox, category


class COCODataset(Dataset):
    def __init__(self, coco_ann_file: str, images_root: str) -> None:
        self.images_root = images_root
        self.coco_ann_file = coco_ann_file
        self.samples_info: List[Dict] = self._read_annotations_for_recognition()

    def _read_annotations_for_recognition(self) -> List[Dict]:
        coco_annotaions: COCO = COCO(self.coco_ann_file)
        img_ids = coco_annotaions.getImgIds()
        samples_info: List[Dict] = []
        for img_id in tqdm(img_ids, desc="Loading COCO annotations...", leave=False):
            image_info: Dict = coco_annotaions.loadImgs(img_id)[0]
            image_relative_path = image_info["file_name"]
            image_path = os.path.join(self.images_root, image_relative_path)
            if not os.path.exists(image_path):
                continue
            ann_ids: List[str] = coco_annotaions.getAnnIds(imgIds=image_info["id"])
            image_annotations: List[Dict] = coco_annotaions.loadAnns(ann_ids)
            for image_annotation in image_annotations:
                sample_info = {
                    "image_path": image_path,
                    "class": image_annotation["category_id"],
                    "bbox": image_annotation["bbox"],
                }
                samples_info.append(sample_info)
        return samples_info

    def _read_annotations_for_detections(self) -> List[Dict]:
        coco_annotaions: COCO = COCO(self.coco_ann_file)
        img_ids = coco_annotaions.getImgIds()
        samples_info: List[Dict] = []
        for img_id in tqdm(img_ids, desc="Loading COCO annotations...", leave=False):
            image_info: Dict = coco_annotaions.loadImgs(img_id)[0]
            image_relative_path = image_info["file_name"]
            image_path = os.path.join(self.images_root, image_relative_path)
            if not os.path.exists(image_path):
                continue
            ann_ids: List[str] = coco_annotaions.getAnnIds(imgIds=image_info["id"])
            image_annotations: List[Dict] = coco_annotaions.loadAnns(ann_ids)
            sample_info = {"image_path": image_path, "objects": image_annotations}
            samples_info.append(sample_info)
        return samples_info

    def __len__(self) -> int:
        return len(self.samples_info)

    def __getitem__(self, index) -> Any:
        return self.samples_info[index]


def extract_datetime_from_filename(filename: str) -> datetime:
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    name = name[len("autosave") :]
    datetime_components = name.split("_")[:-1]
    datetime_components = list(map(int, datetime_components))
    if len(datetime_components) != 6:
        datetime_components.append(0)
    day, month, year, hour, minute, second = datetime_components
    datetime_object = datetime(
        year=year, month=month, day=day, hour=hour, minute=minute, second=second
    )
    return datetime_object


def plot_sample(dataset_sample: Dict) -> None:
    image_path = dataset_sample["image_path"]
    objects = dataset_sample["objects"]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for obj in objects:
        x1, y1, w, h = obj["bbox"]
        image = cv2.rectangle(
            image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), thickness=2
        )
    cv2.imwrite("sample100.png", image[:, :, ::-1])


def get_image_from_recognition_sample(recognition_sample: Dict) -> np.ndarray:
    image = cv2.imread(recognition_sample["image_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x1, y1, w, h = recognition_sample["bbox"]
    # class_category = recognition_sample["class"]
    image_crop = image[y1 : y1 + h, x1 : x1 + w, :]
    return image_crop


def get_class_statistics(dataset: COCODataset) -> None:
    per_class_count: Dict[int, int] = defaultdict(int)
    for sample in tqdm(dataset):
        cls_id = sample["class"]
        per_class_count[cls_id] += 1
    class_encounters: Tuple[int, int] = [
        (cls_id, count) for cls_id, count in per_class_count.items()
    ]
    class_encounters = sorted(class_encounters, key=lambda pair: pair[1], reverse=True)
    pprint(class_encounters)


def get_classes_with_enough_encounters(
    dataset: COCODataset, encounters_thrd: int = 70
) -> List[int]:
    per_class_count: Dict[int, int] = defaultdict(int)
    for sample in tqdm(dataset):
        cls_id = sample["class"]
        per_class_count[cls_id] += 1
    kept_classes: List[int] = []
    for cls_id, count in per_class_count.items():
        if count >= encounters_thrd:
            kept_classes.append(cls_id)
    return kept_classes


def plot_recognition_samples_in_the_grid(
    grid_size: Tuple[int, int], samples: List[Dict]
) -> Figure:
    w, h = grid_size
    fig, axes = plt.subplots(ncols=h, nrows=w)
    for sample_index, sample in enumerate(samples):
        i, j = np.unravel_index(sample_index, shape=(h, w))
        crop = get_image_from_recognition_sample(sample)
        axes[i, j].imshow(crop)
    return fig


def main():
    val_coco_file: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\val_anno.json"
    images_root: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\rtsd-frames"
    dataset = COCODataset(coco_ann_file=val_coco_file, images_root=images_root)
    kept_classes = get_classes_with_enough_encounters(dataset, encounters_thrd=70)
    print(kept_classes)
    # for cls_id in kept_classes:
    #     cls_samples = filter(lambda s: s["class"] == cls_id, dataset.samples_info)
    #     cls_samples = list(cls_samples)[:25]
    #     fig = plot_recognition_samples_in_the_grid(
    #         grid_size=(5, 5), samples=cls_samples
    #     )
    #     fig.savefig(f"CLASS={cls_id}.png")
    #     plt.close(fig)

    # sample = dataset[0]
    # crop = get_image_from_recognition_sample(sample)
    # print(crop.shape)
    # cv2.imwrite("crop.png", crop[:, :, ::-1])


def print_labels_map(classes: List[int]):
    labels_map = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\label_map.json"
    with open(labels_map, "r") as f:
        gost2id = json.load(f)
    id2gost = {v: k for k, v in gost2id.items()}
    gost_encounters = defaultdict(list)
    for cls_id in classes:
        gost = id2gost[cls_id]
        gost_root = gost.split("_")[0]
        gost_encounters[gost_root].append(gost)
    pprint(gost_encounters)


def get_classes_with_sufficient_data():
    val_coco_file: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\val_anno.json"
    train_coco_file: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\train_anno.json"
    images_root: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\rtsd-frames"
    val_dataset = COCODataset(coco_ann_file=val_coco_file, images_root=images_root)
    print(len(val_dataset))
    train_dataset = COCODataset(coco_ann_file=train_coco_file, images_root=images_root)
    train_classes = get_classes_with_enough_encounters(
        train_dataset, encounters_thrd=90
    )
    # pprint(train_classes)
    val_classes = get_classes_with_enough_encounters(val_dataset, encounters_thrd=40)
    # pprint(val_classes)
    taken_classes = set(val_classes).intersection(set(train_classes))
    taken_classes = list(taken_classes)
    pprint("taken_classes")
    print_labels_map(taken_classes)
    print(len(taken_classes))


def explore_seasons_in_dataset() -> None:
    val_coco_file: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\val_anno.json"
    train_coco_file: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\train_anno.json"
    images_root: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\rtsd-frames"
    # dataset = COCODataset(coco_ann_file=val_coco_file, images_root=images_root)
    dataset = COCODataset(coco_ann_file=train_coco_file, images_root=images_root)
    # TODO plot examples of each month
    month_samples = defaultdict(list)
    for sample in tqdm(dataset):
        sample_datetime: datetime = extract_datetime_from_filename(sample["image_path"])
        month = sample_datetime.strftime("%B")
        month_samples[month].append(sample)
    for m, s in month_samples.items():
        print(f"{m}: {len(s)}")
    for month_name, samples in month_samples.items():
        plot_samples = samples[:25]
        fig = plot_recognition_samples_in_the_grid(
            grid_size=(5, 5), samples=plot_samples
        )
        fig.savefig(f"{month_name}.png")
        plt.close(fig)


# TODO try hist Equalization
def dump_october_samples():
    val_coco_file: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\val_anno.json"
    train_coco_file: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\train_anno.json"
    images_root: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\rtsd-frames"
    # dataset = COCODataset(coco_ann_file=val_coco_file, images_root=images_root)
    dataset = COCODataset(coco_ann_file=train_coco_file, images_root=images_root)
    # TODO plot examples of each month
    month_samples = defaultdict(list)
    for sample in tqdm(dataset):
        sample_datetime: datetime = extract_datetime_from_filename(sample["image_path"])
        month = sample_datetime.strftime("%B")
        month_samples[month].append(sample)
    for sample_num, sample in enumerate(month_samples["October"][:25]):
        crop_path = f"October_{sample_num}.png"
        crop = get_image_from_recognition_sample(sample)
        cv2.imwrite(crop_path, crop[:, :, ::-1])


def check_hist_equalization(path: str) -> None:
    bgr_image: np.ndarray = cv2.imread(path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    eq_img = cv2.equalizeHist(gray_img)
    fig, (rgb_ax, gray_ax) = plt.subplots(1, 2)
    rgb_ax.imshow(rgb_image)
    gray_ax.imshow(eq_img, cmap="gray")
    path_basename = os.path.basename(path)
    path_basename = os.path.splitext(path_basename)[0]
    output_path = f"{path_basename}_Equalized.png"
    pprint(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def draw_equalized():
    root: str = "E:\\ITMO\\TrafficSignsCV"
    paths = os.listdir(root)
    paths = list(filter(lambda p: "October" in p, paths))
    paths = sorted(paths)
    for p in paths:
        im_path = os.path.join(root, p)
        check_hist_equalization(im_path)


if __name__ == "__main__":
    draw_equalized()
