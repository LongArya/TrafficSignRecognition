from PIL import Image
import torch
from typing import Any
from copy import deepcopy
from typing import Callable
from torch.utils.data import Dataset


class TransformApplier(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transformation: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.dataset = dataset
        self.transformation = transformation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        """
        Required keys:
            - image
        """
        sample = deepcopy(self.dataset[index])
        sample["image"] = self.transformation(sample["image"])
        return sample
