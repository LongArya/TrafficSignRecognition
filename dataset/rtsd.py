from PIL import Image
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


class RussianTrafficSignBaseDataset(Dataset):
    def __init__(
        self, coco_ann_file: str, images_root: str, label_map_file: str
    ) -> None:
        self.images_root = images_root
        self.coco_ann_file = coco_ann_file
        with open(label_map_file, "r") as f:
            name2id_mapping = json.load(f)
            self.id2name_mapping = {id: name for name, id in name2id_mapping.items()}
        self.samples_info: List[Dict] = self._read_annotations_for_recognition()

    @staticmethod
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
            sample_datetime = (
                RussianTrafficSignBaseDataset.extract_datetime_from_filename(image_path)
            )
            for image_annotation in image_annotations:
                class_id = image_annotation["category_id"]
                sample_info = {
                    "image_path": image_path,
                    "class": class_id,
                    "class_name": self.id2name_mapping[class_id],
                    "bbox": image_annotation["bbox"],
                    "datetime": sample_datetime,
                }
                samples_info.append(sample_info)
        return samples_info

    def __len__(self) -> int:
        return len(self.samples_info)

    def __getitem__(self, index) -> Any:
        return self.samples_info[index]


class SquareCropReader(Dataset):
    def __init__(self, base_dataset: RussianTrafficSignBaseDataset) -> None:
        self._base_dataset = base_dataset
        super().__init__()

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, index: int) -> Dict:
        base_sample = self._base_dataset[index]
        image: np.ndarray = cv2.imread(base_sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        x1, y1, w, h = base_sample["bbox"]
        xc = x1 + w // 2
        yc = y1 + h // 2
        crop_size: int = max(w, h)

        crop_x1 = xc - crop_size // 2
        crop_x2 = xc + crop_size // 2
        crop_y1 = yc - crop_size // 2
        crop_y2 = yc + crop_size // 2
        crop_x1 = np.clip(crop_x1, 0, image_width - 1)
        crop_x2 = np.clip(crop_x2, 0, image_width - 1)
        crop_y1 = np.clip(crop_y1, 0, image_height - 1)
        crop_y2 = np.clip(crop_y2, 0, image_height - 1)

        image_crop = image[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1, :]
        base_sample["image"] = Image.fromarray(image_crop)

        return base_sample
