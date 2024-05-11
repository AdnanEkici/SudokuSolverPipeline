from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from dnet import Dnet
from sudoku_digit_dataset import SudokuDigitDataset
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.logger import Logger  # noqa
import utils.utils as utils  # noqa


def inference_with_visualization(model: Dnet, test_data_path: str | None, device: str = "cpu"):
    test_data_path = test_data_path
    test_dataset = SudokuDigitDataset(root_dir=test_data_path, transform=Dnet.transform, logger=logger)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            # Visualize predictions
            for i in range(inputs.size(0)):
                image = inputs[i].cpu().numpy().squeeze()
                image = (image * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = cv2.resize(image, (600, 600))
                cv2.putText(image, f"Label: {utils.label_mapper(labels[i].item())}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, f"Prediction: {utils.label_mapper(predicted[i].item())}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("Prediction", image)
                cv2.waitKey(0)  # Wait for a key press to continue
                cv2.destroyAllWindows()
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start DNET model training with selected config.")
    parser.add_argument(
        "-cp",
        "--configuration_file_path",
        help="Path to the traing YAML file. DEFAULT=app/dnet/dnet_configuration_file.yml",
        default="app" + os.sep + "dnet" + os.sep + "dnet_configuration_file.yml",
        required=False,
    )  # noqa
    parser.add_argument(
        "-lm", "--logger_name", help="Logger file name for specific experiment. DEFAULT=dnet_trainer.log", default="dnet_trainer.log", required=False
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug mode. DEFAULT=False")
    args = parser.parse_args()

    log_file_name = args.logger_name
    config_path = args.configuration_file_path
    verbose_mode = args.verbose
    logger = Logger(log_file=log_file_name, debug_mode=verbose_mode)

    config = utils.read_yaml(config_path, logger=logger)
    if "inference" not in config:
        raise KeyError("Your configuration file must contain inference section !")
    config = config["inference"]

    test_data_path: str | None = config.get("test_dataset_path", None)
    device: str = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    pre_trained_weights: str | None = config.get("pre_trained_weights", None)

    logger.info("Dnet Inference started.")
    logger.info(f"Test dataset path: {test_data_path}")
    logger.info(f"Selected device: {device}")
    logger.info(f"Pre-trained weights will loaded from : {pre_trained_weights}")
    logger.info("Verbose mode is " + ("online" if verbose_mode else "offline"))

    model = Dnet()
    model.to(device)
    model.load_state_dict(torch.load(pre_trained_weights))
    model.eval()
    test_predictions = inference_with_visualization(model=model, test_data_path=test_data_path, device=device)
