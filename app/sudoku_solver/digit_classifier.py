import cv2
import torch
import os
import sys
from typing import Union
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.logger import Logger  # noqa
import utils.utils as utils  # noqa
import dnet.dnet as dnet  # noqa

# TODO add logger

class DigitClassifier:
    def __init__(self, model_path: Union[str, None] = None, device: Union[str, None] = None):
        self.model = dnet.Dnet()
        self.model_path = model_path
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Classifier using: {self.device}")
        self.transform = dnet.Dnet.transform
        self.__load_model()

    def classify_digit(self, digit):
        image = self.transform(digit)
        image_tensor = image.unsqueeze(0).to(self.device)
        output = self.model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = utils.label_mapper(predicted.item())
        print(f"Prediction: {predicted_label}")
        return predicted_label

    def __load_model(self):
        print("Loading DNET model.")
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

