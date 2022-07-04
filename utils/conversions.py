# Copyright 2022 Daniele Rege Cambrin
from itertools import product, groupby
from typing import List, Tuple
import torch
import torchvision.transforms.functional as TF

Position = Tuple[int, int]


def extract_rgb(img: torch.Tensor, rgb_channels: Tuple[int, int, int]) -> torch.Tensor:
    return (img[rgb_channels, :] * 255).round().byte()


def crop_image(
    image: torch.Tensor, crop_size: int
) -> Tuple[torch.Tensor, List[Position]]:
    cropped_images = []
    crop_position = []
    for top, left in product(
        range(0, image.size()[-1], crop_size),
        range(0, image.size()[-2], crop_size),
    ):
        cropped_images.append(
            TF.crop(image, top, left, crop_size, crop_size).unsqueeze(0)
        )
        crop_position.append((top, left))
    return torch.concat(cropped_images), crop_position


def recompose_image(crops: torch.Tensor, positions: List[Position]) -> torch.Tensor:
    zipped = sorted(zip(positions, crops), key=lambda x: x[0])
    rows = [
        torch.concat([el[1] for el in g], dim=2)
        for k, g in groupby(zipped, key=lambda x: x[0][0])
    ]
    return torch.concat(rows, dim=1)
