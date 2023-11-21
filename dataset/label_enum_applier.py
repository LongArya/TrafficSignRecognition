from torch.utils.data import Dataset
from copy import deepcopy
from typing import List, Dict


BG_ID = 0
BG_CLASS = "background"


class LabelEnumApplier(Dataset):
    def __init__(self, base_dataset: Dataset, label_enum_id2name: Dict[int, str]):
        self._base_dataset = base_dataset
        self.label_enum_id2name = label_enum_id2name
        self.label_enum_name2id = {name: id for id, name in label_enum_id2name.items()}

    def __getitem__(self, index: int) -> Dict:
        sample = self._base_dataset[index]
        sample_class_name = sample["class_name"]
        updated_id = self.label_enum_name2id.get(sample_class_name, BG_ID)
        if updated_id == BG_ID:
            sample["class_name"] = BG_CLASS
        sample["class"] = updated_id
        return sample
