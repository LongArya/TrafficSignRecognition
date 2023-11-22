import torch
from pprint import pprint
from typing import Tuple, List
import matplotlib.pyplot as plt
import onnxruntime
import numpy as np
import cv2
from scipy.special import softmax
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_IMAGE = os.path.join(SCRIPT_DIR, "assets", "example_image.png")
EXAMPLE_XYWH_BBOX = [622, 375, 30, 27]
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")
RESNET18_PATH_LABELS = (
    os.path.join(MODELS_ROOT, "resnet18", "traffic_sign_recognition20.onnx"),
    os.path.join(MODELS_ROOT, "resnet18", "traffic_label_enum.json"),
)
RESNET34_PATH_LABELS = (
    os.path.join(MODELS_ROOT, "resnet34", "traffic_sign_recognition21.onnx"),
    os.path.join(MODELS_ROOT, "resnet34", "traffic_label_enum.json"),
)


class ONNXTrafficSignClassifier:
    """Class for encapsulating inference of model trained for traffic sign classification via ONNX"""

    def __init__(self, model_path: str, label_id2name_file: str):
        self.ort_sess = onnxruntime.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )
        with open(label_id2name_file, "r") as f:
            self._label_id2name = json.load(f)
        self._label_id2name = {int(k): v for k, v in self._label_id2name.items()}
        self._input_size = (224, 224)
        self._imagenet_mean = np.array([0.485, 0.456, 0.406])
        self._imagenet_std = np.array([0.229, 0.224, 0.225])

    def _preprocessing(self, input_image: np.ndarray) -> np.ndarray:
        network_input = input_image.astype(np.float32)
        network_input = cv2.resize(network_input, self._input_size)
        network_input /= 255
        network_input -= self._imagenet_mean
        network_input /= self._imagenet_std
        network_input = np.transpose(network_input, (2, 0, 1))
        network_input = np.expand_dims(network_input, 0)
        return network_input

    def _parse_model_output(self, model_logits: np.ndarray) -> Tuple[str, float]:
        probs = softmax(model_logits)
        pred_class_index = np.argmax(probs)
        prediction_probability: float = probs[pred_class_index]
        predicted_sign: str = self._label_id2name[pred_class_index]
        return predicted_sign, prediction_probability

    def _prepare_square_crop(
        self, image: np.ndarray, bbox_xyxy: List[float]
    ) -> Tuple[str, float]:
        x1, y1, x2, y2 = bbox_xyxy
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        w = x2 - x1
        h = y2 - y1
        image_height, image_width = image.shape[:2]
        crop_size: int = max(w, h)
        crop_x1 = xc - crop_size // 2
        crop_x2 = xc + crop_size // 2
        crop_y1 = yc - crop_size // 2
        crop_y2 = yc + crop_size // 2
        crop_x1 = np.clip(crop_x1, 0, image_width - 1)
        crop_x2 = np.clip(crop_x2, 0, image_width - 1)
        crop_y1 = np.clip(crop_y1, 0, image_height - 1)
        crop_y2 = np.clip(crop_y2, 0, image_height - 1)
        crop = image[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1, :]
        return crop

    def __call__(self, image: np.ndarray, bbox_xyxy: List[float]) -> Tuple[str, float]:
        """Classifies part of the image inside bbox, returns name of the predicted class and score"""
        crop = self._prepare_square_crop(image, bbox_xyxy)
        network_input = self._preprocessing(crop)
        model_logits = self.ort_sess.run(None, {"input.1": network_input})[0][0]
        predicted_sign, prediction_probability = self._parse_model_output(model_logits)
        return predicted_sign, prediction_probability


def example_onnx_infer(onnx_model_path: str, label_enum_file: str):
    classifier = ONNXTrafficSignClassifier(
        model_path=onnx_model_path, label_id2name_file=label_enum_file
    )
    img = cv2.imread(EXAMPLE_IMAGE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x1, y1, w, h = EXAMPLE_XYWH_BBOX
    bbox_xyxy = [x1, y1, x1 + w, y1 + h]
    output = classifier(img, bbox_xyxy)
    pprint(output)


if __name__ == "__main__":
    example_onnx_infer(*RESNET18_PATH_LABELS)
    example_onnx_infer(*RESNET34_PATH_LABELS)
