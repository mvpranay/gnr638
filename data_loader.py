import time
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import APDNN
import random


class Data_Loader:
    """Custom Dataset for loading images and converting to framework tensors."""

    def __init__(
        self,
        data_dir: str,
        img_size: int = 32,
        transform=None,
        print_time: bool = False,
    ):
        """
        Args:
            data_dir: Path to dataset folder with class subdirectories
            img_size: Target image size (default 32x32)
            transform: Optional transforms to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_idx = {}
        self.print_time = print_time

        self.load_time = 0.0
        self._load_data()

    def _load_data(self):
        """Load image paths and labels from directory structure."""
        start_time = time.time()

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory {self.data_dir} does not exist")

        # Get class labels from subdirectories
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if len(class_dirs) == 0:
            raise ValueError(f"No class subdirectories found in {self.data_dir}")

        for idx, class_dir in enumerate(class_dirs):
            label = class_dir.name
            self.label_to_idx[label] = idx

            img_count = 0
            for img_file in sorted(class_dir.glob("*.png")):
                self.images.append(str(img_file))
                self.labels.append(label)
                img_count += 1

            print(f"  Class '{label}': {img_count} images")

        self.load_time = time.time() - start_time
        if self.print_time:
            print(f"\nDataset loading time: {self.load_time:.4f} seconds")
        print(f"Total images: {len(self.images)}")
        print(f"Classes: {list(self.label_to_idx.keys())}\n")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[list, int]:
        """
        Return image as list and label as integer.

        Returns:
            (image_array, label_idx) where image_array is 32x32x3 normalized to [0,1]
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]

        # Load and resize image using OpenCV
        image = cv2.imread(img_path)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image
        image = cv2.resize(
            image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )

        # Convert to list and normalize to [0, 1]
        # OpenCV returns numpy array, convert to nested list [H, W, C]
        img_array_3d = []
        for h in range(self.img_size):
            row = []
            for w in range(self.img_size):
                pixel = [float(image[h, w, c]) / 255.0 for c in range(3)]
                row.append(pixel)
            img_array_3d.append(row)
        
        # Apply transform if provided
        if self.transform:
            img_array_3d = self.transform(img_array_3d)

        return img_array_3d, label_idx

    def to_tensor(self, img_array: list) -> APDNN.Tensor:
        """Convert list array to framework Tensor (flattened for now)."""
        # Flatten: 32x32x3 -> 3072
        flat_data = []
        for row in img_array:
            for pixel in row:
                for channel_val in pixel:
                    flat_data.append(channel_val)
        return APDNN.Tensor(flat_data, [1, len(flat_data)])
    
    def get_batch(self, indices: List[int]) -> Tuple[List, List[int]]:
        """
        Get a batch of images and labels.
        
        Returns:
            (list of framework tensors, list of label indices)
        """
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            img_array, label_idx = self[idx]
            tensor = self.to_tensor(img_array)
            batch_images.append(tensor)
            batch_labels.append(label_idx)
        
        return batch_images, batch_labels
    
    def get_class_names(self) -> List[str]:
        """Return list of class names."""
        return list(self.label_to_idx.keys())


def get_data_loader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    img_size: int = 32,
    transform=None,
    test_split: float = 0.0,
    print_time: bool = False
) -> Tuple[Data_Loader, Optional[Data_Loader]]:
    """
    Create Data_Loader object(s) for dataset.
    
    Args:
        data_dir: Path to dataset folder
        batch_size: Batch size for loading
        shuffle: Whether to shuffle data
        img_size: Target image size
        transform: Optional image transforms
        test_split: If > 0, split data into train/test
    
    Returns:
        If test_split == 0: (dataset, None)
        If test_split > 0: (train_dataset, test_dataset)
    """
    print(f"Loading dataset from {data_dir}...")
    dataset = Data_Loader(data_dir, img_size=img_size, transform=transform, print_time=print_time)
    
    if test_split > 0:
        train_size = int(len(dataset) * (1 - test_split))
        test_size = len(dataset) - train_size
        
        # Split indices
        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_dataset = _DatasetSplit(dataset, train_indices)
        test_dataset = _DatasetSplit(dataset, test_indices)
        
        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    return dataset, None


class _DatasetSplit:
    """Wrapper for dataset splits."""
    
    def __init__(self, dataset: Data_Loader, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
        self.load_time = dataset.load_time
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]
    
    def get_batch(self, batch_indices: List[int]):
        actual_indices = [self.indices[i] for i in batch_indices]
        return self.dataset.get_batch(actual_indices)
    
    def get_class_names(self):
        return self.dataset.get_class_names()