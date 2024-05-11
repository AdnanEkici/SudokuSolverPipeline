from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
from torch.utils.data import DataLoader


class SudokuDigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(self.class_to_idx)
        self.images = self.make_dataset()


    def make_dataset(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append((img_path, self.class_to_idx[cls]))

        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        # Load image using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        return image, label

# # Initialize the dataset
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
# ])

# train_data_path = r'C:\Users\Pc\Desktop\Sudoku-Dataset\extracted_digit_data\train'


# # Loop through the dataset
# for image, label in dataset:
#     # Convert tensor to numpy array and transpose it to (H, W, C)
#     image_np = image.permute(1, 2, 0).numpy()
#     # Convert numpy array to OpenCV's format (BGR)
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     # Display the image
#     cv2.imshow('Image', image_bgr)
#     # Set the title of the image window to the label
#     cv2.setWindowTitle('Image', str(label))
#
#     # Wait for a key press and close the window if 'q' is pressed
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
#
# # Close all OpenCV windows
# cv2.destroyAllWindows()

# SHOW BATCH
# import matplotlib.pyplot as plt
# import numpy as np
# def show_images(images, labels, ncols=8, figsize=(15, 5)):
#     nrows = int(np.ceil(len(images) / ncols))
#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
#     for i, ax in enumerate(axes.flat):
#         if i < len(images):
#             ax.imshow(images[i], cmap='gray')
#             ax.set_title(f"Label: {labels[i]}")
#             ax.axis('off')
#         else:
#             ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
# train_dataset = SudokuDigitDataset(root_dir=train_data_path, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# # Assuming `images` and `labels` are obtained from the DataLoader
# # You can directly iterate through the DataLoader as shown below
# for i, (images, labels) in enumerate(train_loader):
#     # Visualize the batch of images and labels
#     show_images(images.squeeze(), labels)
#     break  # Break after displaying one batch for demonstration purposes
