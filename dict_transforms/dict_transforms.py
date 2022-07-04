# Copyright 2022 Daniele Rege Cambrin
import math
from typing import Callable, Any, Dict, Union, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
import random
import skimage.transform as ST


def tensor_image_to_ndarray(img: torch.Tensor) -> np.ndarray:
    img = img.numpy()
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    return img


def forward_dict(
    x: Dict[str, Any],
    transform: Callable[[Union[torch.Tensor, np.ndarray]], torch.Tensor],
):
    for k, v in x.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            x[k] = transform(v)
        else:
            x[k] = v
    return x


class ToTensor(torch.nn.Module):
    def forward(self, x: Dict[str, Any]):
        return forward_dict(x, TF.to_tensor)


class RandomHorizontalFlipping(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Dict[str, Any]):
        return forward_dict(
            x, TF.hflip if random.uniform(0.0, 1.0) < self.p else lambda img: img
        )


class RandomVerticalFlipping(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Dict[str, Any]):
        return forward_dict(
            x, TF.vflip if random.uniform(0.0, 1.0) < self.p else lambda img: img
        )


class RandomRotate(torch.nn.Module):
    def __init__(self, p: float, rotation: Union[float, Tuple[float, float]]):
        super().__init__()
        self.p = p

        if not isinstance(rotation, tuple):
            rotation = (-rotation, rotation)
        self.rotation = rotation

    def forward(self, x: Dict[str, Any]):
        rotation = random.uniform(self.rotation[0], self.rotation[1])

        def rotate(img: torch.Tensor):
            rotated = ST.rotate(
                tensor_image_to_ndarray(img),
                rotation,
                preserve_range=True,
                mode="reflect",
            )
            return TF.to_tensor(rotated)

        return forward_dict(
            x, rotate if random.uniform(0.0, 1.0) < self.p else lambda img: img
        )


class RandomShear(torch.nn.Module):
    def __init__(self, p: float = 0.5, shear: Union[float, tuple] = 20):
        super().__init__()
        self.p = p
        if not isinstance(shear, tuple):
            shear = (-shear, shear)
        self.shear = [math.radians(x) for x in shear]

    def forward(self, x: Dict[str, Any]):
        angle = self.shear[0] + (self.shear[1] - self.shear[0]) * random.random()
        tr = ST.AffineTransform(shear=angle)

        def shear(img):
            warped = ST.warp(
                tensor_image_to_ndarray(img), tr, mode="reflect", preserve_range=True
            )
            return TF.to_tensor(warped)

        return forward_dict(
            x, shear if random.uniform(0.0, 1.0) < self.p else lambda img: img
        )


class ConvertToFloat(torch.nn.Module):
    def forward(self, x: Dict[str, Any]):
        return forward_dict(x, TF.convert_image_dtype)
