# Some parts of this code is assisted by AI Agent

import h5py
import os

import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader


class BuildDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        data_paths = {fname.split("_")[2]: f"{data_dir}/{fname}" for fname in os.listdir(data_dir)}

        ################################################################################################################
        # TODO: Initialize the dataset. Whether you do operations in __init__ or __getitem__ is flexible as long as you
        #  pass the assertions
        ################################################################################################################
        # Load images from h5 file (key is 'data', not 'images')
        with h5py.File(data_paths["img"], 'r') as f:
            self.images = f['data'][:]  # Shape: (N, 3, 300, 400)
        
        # Load labels (each element is an array of labels for that image)
        self.labels = np.load(data_paths["labels"], allow_pickle=True)  # Shape: (N,) where each element is array
        
        # Load bounding boxes (each element is an array of bboxes for that image)
        self.bboxes = np.load(data_paths["bboxes"], allow_pickle=True)  # Shape: (N,) where each element is array
        
        # Load masks from h5 file
        with h5py.File(data_paths["mask"], 'r') as f:
            all_masks = f['data'][:]  # Shape: (total_objects, 300, 400)
        
        # Build mapping from image index to mask indices
        # The masks are stored sequentially, so we need to map them back to images
        self.masks = []
        mask_idx = 0
        
        for i in range(len(self.labels)):
            num_objects = len(self.labels[i])
            image_masks = all_masks[mask_idx:mask_idx + num_objects]
            self.masks.append(image_masks)
            mask_idx += num_objects
        
        # Convert images to float and normalize to [0, 1]
        self.images = self.images.astype(np.float32) / 255.0
        
        # Define transforms
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((800, 1066)),  # Rescale to 800x1066
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Dataset initialized:")
        print(f"  Total images: {self.images.shape[0]}")
        print(f"  Image shape: {self.images.shape[1:]}")
        print(f"  Total mask objects: {len(all_masks)}")
        print(f"  Labels per image sample: {[len(l) for l in self.labels[:5]]}")
        ################################################################################################################

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ################################################################################################################
        # TODO: Implement __getitem__. Whether you do operations in __init__ or __getitem__ is flexible as long as you
        #  pass the assertions
        ################################################################################################################
        # Get image and convert to tensor
        img = torch.from_numpy(self.images[index])  # Shape: (3, 300, 400)
        
        # Apply transforms: resize to 800x1066, then normalize, then pad to 800x1088
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(800, 1066), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # Normalize
        img = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )(img)
        
        # Zero pad to 800x1088
        transformed_img = torch.nn.functional.pad(img, (0, 22, 0, 0))  # pad right by 22 pixels
        
        # Get labels
        labels = torch.from_numpy(self.labels[index].astype(np.int64))
        
        # Get masks and transform them
        masks = self.masks[index]  # Shape: (n_obj, 300, 400)
        masks = torch.from_numpy(masks.astype(np.float32))
        
        # Resize masks to match image size (800x1066)
        if len(masks) > 0:
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1), size=(800, 1066), mode='bilinear', align_corners=False
            ).squeeze(1)
            # Pad masks to 800x1088
            masks = torch.nn.functional.pad(masks, (0, 22, 0, 0))
            # Add channel dimension
            transformed_masks = masks.unsqueeze(1)  # Shape: (n_obj, 1, 800, 1088)
        else:
            transformed_masks = torch.empty(0, 1, 800, 1088)
        
        # Get bounding boxes and transform them
        bboxes = torch.from_numpy(self.bboxes[index].astype(np.float32))
        
        # Scale bounding boxes to match the resized image
        if len(bboxes) > 0:
            # Original image size: 300x400, new size: 800x1066 (then padded to 800x1088)
            scale_y = 800.0 / 300.0
            scale_x = 1066.0 / 400.0
            
            transformed_bboxes = bboxes.clone()
            transformed_bboxes[:, [0, 2]] *= scale_x  # x coordinates
            transformed_bboxes[:, [1, 3]] *= scale_y  # y coordinates
        else:
            transformed_bboxes = torch.empty(0, 4)
        ################################################################################################################

        # Check flag
        n_obj = labels.shape[0]
        assert transformed_img.shape == (3, 800, 1088,), f"Expected transformed_img shape {(3, 800, 1088,)}, but got {transformed_img.shape}."
        assert labels.shape == (n_obj,), f"Expected labels shape {(n_obj,)}, but got {labels.shape}."
        assert transformed_masks.shape == (n_obj, 1, 800, 1088,), f"Expected transformed_masks shape {(n_obj, 1, 800, 1088,)}, but got {transformed_masks.shape}."
        assert transformed_bboxes.shape == (n_obj, 4,), f"Expected transformed_bboxes shape {(n_obj, 4,)}, but got {transformed_bboxes.shape}."

        return transformed_img, labels, transformed_masks, transformed_bboxes

    def __len__(self):
        return self.images.shape[0]


class BuildDataLoader(DataLoader):
    def __init__(self, dataset: BuildDataset, **kwargs):
        kwargs["collate_fn"] = BuildDataLoader.collate_fn
        super().__init__(dataset=dataset, **kwargs)

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        ################################################################################################################
        # TODO: Implement collate_fn to load batches
        # Hint: Which data are stackable and which are not?
        ################################################################################################################
        # Separate the batch into individual components
        imgs, labels, masks, bboxes = zip(*batch)
        
        # Images can be stacked since they all have the same shape
        imgs = torch.stack(imgs, dim=0)  # Shape: (batch_size, 3, 800, 1088)
        
        # Labels, masks, and bboxes cannot be stacked because each image has different number of objects
        # So we keep them as lists
        labels = list(labels)  # List of tensors, each with shape (n_obj_i,)
        masks = list(masks)    # List of tensors, each with shape (n_obj_i, 1, 800, 1088)
        bboxes = list(bboxes)  # List of tensors, each with shape (n_obj_i, 4)
        ################################################################################################################

        return imgs, labels, masks, bboxes




