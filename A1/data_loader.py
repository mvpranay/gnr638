import os
import time
import random
import cv2
import APDNN

IMG_HEIGHT = 32
IMG_WIDTH = 32


class ImageFolderDataset:
    """
    Custom dataset loader for:
    - Any folder structured datasets with string labels
    - Converts images to grayscale
    - Resizes to IMG_HEIGHT x IMG_WIDTH
    - Returns batches as APDNN.Tensor
    - Maps string labels to integer indices
    - Measures disk loading time
    """

    def __init__(self, data_dir, label_to_idx=None, batch_size=32, shuffle=True, augment=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.label_to_idx = label_to_idx  # Map string label -> int

        self.samples = []   # (filepath, label_idx)
        self._index_dataset()

    # ---------------------------------------------------------
    # Index dataset (scan folder structure)
    # ---------------------------------------------------------
    def _index_dataset(self):
        # Build a label mapping for non-numeric folder names if none is provided
        if self.label_to_idx is None:
            folder_names = [
                name for name in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, name))
            ]

            # If any folder name is non-numeric, map all folder names to indices
            has_non_numeric = False
            for name in folder_names:
                try:
                    int(name)
                except ValueError:
                    has_non_numeric = True
                    break

            if has_non_numeric:
                self.label_to_idx = {name: idx for idx, name in enumerate(sorted(folder_names))}

        for label_name in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label_name)

            if not os.path.isdir(label_path):
                continue

            # Try to get label index from mapping, or parse as int
            if self.label_to_idx is not None:
                if label_name not in self.label_to_idx:
                    continue
                label = self.label_to_idx[label_name]
            else:
                try:
                    label = int(label_name)
                except ValueError:
                    continue

            for file_name in os.listdir(label_path):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(label_path, file_name)
                    self.samples.append((full_path, label))

    # ---------------------------------------------------------
    # Optional simple augmentation (no numpy)
    # ---------------------------------------------------------
    def _augment_image(self, img):
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Random small rotation (-10 to +10 degrees)
        angle = random.uniform(-10, 10)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

        return img

    # ---------------------------------------------------------
    # Iterator for batching
    # ---------------------------------------------------------
    def __iter__(self):

        if self.shuffle:
            random.shuffle(self.samples)

        loading_time = 0

        for i in range(0, len(self.samples), self.batch_size):
            start = time.time()
            
            batch = self.samples[i:i + self.batch_size]

            flat_batch_data = []
            labels = []

            for filepath, label in batch:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Resize to 32x32
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

                if self.augment:
                    img = self._augment_image(img)

                # Flatten + normalize manually (no numpy)
                normalized_flat = (img.reshape(-1) / 255.0).astype("float32").tolist()
                flat_batch_data.extend(normalized_flat)

                labels.append(label)

            if len(labels) == 0:
                continue

            # Construct tensor [B, 1, IMG_HEIGHT, IMG_WIDTH]
            tensor_x = APDNN.Tensor(
                flat_batch_data,
                [len(labels), 1, IMG_HEIGHT, IMG_WIDTH]
            )
            
            end = time.time()
            loading_time += end - start

            yield tensor_x, labels

        print(f"Total dataset loading time: {loading_time:.4f} seconds")

    # ---------------------------------------------------------
    # Dataset length
    # ---------------------------------------------------------
    def __len__(self):
        return len(self.samples)
