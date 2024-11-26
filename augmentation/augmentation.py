import albumentations as aug

from albumentations import Compose


def get_train_transform(size: int) -> Compose:
    transforms = [
        aug.Resize(height=size, width=size),
        aug.Downscale(interpolation=0, scale_min=0.15, scale_max=0.8, p=1),
        aug.OneOf([
            aug.RandomRotate90(p=1),
            aug.HorizontalFlip(p=1),
            aug.VerticalFlip(p=1),
        ], p=0.25),
    ]
    transform = aug.Compose(
        transforms=transforms,
        bbox_params=aug.BboxParams(format='pascal_voc', label_fields=['labels'])
    )
    return transform


def get_valid_transform(size: int) -> Compose:
    transforms = [
        aug.Resize(height=size, width=size),
        aug.Downscale(interpolation=0, scale_min=0.15, scale_max=0.8, p=1),
    ]
    transform = aug.Compose(
        transforms=transforms,
        bbox_params=aug.BboxParams(format='pascal_voc', label_fields=['labels'])
    )
    return transform
