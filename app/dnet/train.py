from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from dnet import Dnet
from sudoku_digit_dataset import SudokuDigitDataset  # noqa
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.logger import Logger  # noqa
import utils.utils as utils  # noqa


def validate_model(
    model: Dnet, validation_loader: torch.utils.data.dataloader.DataLoader, criterion: torch.nn.modules.loss.CrossEntropyLoss, device: str = "cpu"
):
    model.eval()  # Set model to evaluation mode
    validation_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():  # No need to compute gradients during validation
        for data in validation_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate validation loss and accuracy
    validation_loss /= len(validation_loader)
    accuracy = correct_predictions / total_samples * 100.0

    # Log validation metrics
    logger.info(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return validation_loss, accuracy


# Todo add pre-training
def train_model(
    train_dataset_path: str | None,
    validation_dataset_path: str | None,
    device: str = "cpu",
    number_of_epochs: int = 100,
    model_save_path: str = "dnet_saved_models",
    save_frequency: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    enable_augmentations: bool = True,
):
    model = Dnet()
    train_dataset = SudokuDigitDataset(
        root_dir=train_dataset_path, transform=Dnet.transform, logger=logger, enable_augmentations=enable_augmentations
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate = False
    if validation_dataset_path is not None and validation_dataset_path != "":
        validation_dataset = SudokuDigitDataset(root_dir=validation_dataset_path, transform=Dnet.transform, logger=logger, enable_augmentations=False)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        validate = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in tqdm(range(number_of_epochs), colour="CYAN"):
        running_loss = 0.0
        for _, data in enumerate(train_loader, start=0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        validate and validate_model(model=model, validation_loader=validation_loader, criterion=criterion, device=device)
        # Save the model
        if (epoch + 1) % save_frequency == 0:
            torch.save(model.state_dict(), model_save_path + os.sep + f"epoch{epoch + 1}.pth")
            logger.info(f"Model saved to {model_save_path} named as epoch{epoch + 1}.pth")

    # Save the model after the last epoch
    torch.save(model.state_dict(), model_save_path + os.sep + "last.pth")
    logger.info(f"Final model saved at epoch {number_of_epochs}")


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

    if "training" not in config:
        raise KeyError("Your configuration file must contain training section !")
    config = config["training"]

    train_dataset_path: str | None = config.get("train_dataset_path", None)
    validation_dataset_path: str | None = config.get("validation_dataset_path", None)
    device: str = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    number_of_epochs: int = config.get("number_of_epochs", 100)
    save_frequency: int = config.get("save_frequency", 5)
    pre_trained_weights: str | None = config.get("pre_trained_weights", None)
    batch_size: int = config.get("batch_size", 64)
    model_save_path: str = config.get("model_save_path", "dnet_saved_models")
    learning_rate: float = config.get("learning_rate", 0.001)
    enable_augmentations: bool = config.get("enable_augmentations", False)

    logger.info("Dnet Training started.")
    logger.info(f"Train dataset path: {train_dataset_path}")
    (logger.info(f"Validation dataset path: {validation_dataset_path}")) if validation_dataset_path is not None else (
        logger.warning("No validation path specified skipping validation.")
    )
    logger.info(f"Selected device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Number of epochs: {number_of_epochs}")
    logger.info(f"Model save frequency: {save_frequency}")
    logger.info(f"Model save path is: {model_save_path}")
    pre_trained_weights is not None and logger.info(f"Pre-trained weights will loaded from : {pre_trained_weights}")
    logger.info("Verbose mode is " + ("online" if verbose_mode else "offline"))
    logger.info("Augmentations are: " + ("ON" if enable_augmentations else "OFF"))

    train_model(
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        device=device,
        model_save_path=model_save_path,
        number_of_epochs=number_of_epochs,
        save_frequency=save_frequency,
        learning_rate=learning_rate,
        batch_size=batch_size,
        enable_augmentations=enable_augmentations,
    )
