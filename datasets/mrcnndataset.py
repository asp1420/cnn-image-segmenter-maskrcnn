import os
import torch
import numpy as np

from scipy.ndimage import label
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from albumentations import Compose
from torch import Tensor


class MRCNNDataset(Dataset):

    def __init__(self, root: int, num_channels: int, transforms: Compose=None) -> None:
        self.root = root
        self.num_channels = num_channels
        self.transforms = transforms
        self.masks = list(sorted(os.listdir(join(root, "masks"))))

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        img_path = os.path.join(self.root, "images", self.masks[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = np.load(img_path)[..., :self.num_channels]
        max_size = np.max(img.shape)
        mask = np.load(mask_path)
        ids = np.unique(mask)[1:]
        masks = np.zeros((1, ) + mask.shape) > 0
        boxes = list()
        for obj_id in ids:
            mask_id = np.where(mask == obj_id, 1, 0)
            mask_labels, _ = label(mask_id)
            obj_ids = np.unique(mask_labels)
            obj_ids = obj_ids[1:]
            masks_id = mask_labels == obj_ids[:, None, None]
            num_objs = len(obj_ids)
            for i in range(num_objs):
                pos = np.where(masks_id[i])
                x_min = np.min(pos[1])
                x_max = np.max(pos[1])
                y_min = np.min(pos[0])
                y_max = np.max(pos[0])
                boxes.append([x_min, y_min, x_max, y_max])
            masks = np.append(masks, masks_id, axis=0)

        masks = masks[1:]
        masks = np.transpose(masks, (1, 2, 0))
        # Clips boxes from 0 to max crop size
        boxes = np.array(boxes)
        boxes = np.clip(boxes, a_min=0., a_max=max_size)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.tensor(np.ones(len(boxes)), dtype=torch.int64)
        image_id = torch.tensor([idx])
        is_crowd = torch.zeros(len(labels), dtype=torch.int64)

        target = dict()
        target.update({'labels': labels})
        target.update({'image_id': image_id})
        target.update({'iscrowd': is_crowd})

        if self.transforms is not None:
            sample = {
                'image': img,
                'bboxes': boxes,
                'labels': target['labels'],
                'mask': np.array(masks).astype(np.uint8)
            }
            sample = self.transforms(**sample)
            img = to_tensor(sample['image'])
            boxes = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
            masks = to_tensor(sample['mask'].astype(bool))
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target.update({'boxes': boxes})
            target.update({'area': area})
            target.update({'masks': masks})
        return img, target

    def __len__(self) -> int:
        return len(self.masks)
