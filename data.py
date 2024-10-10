import os

import torch
import torchvision.io as io

class CityScapesDataset(torch.utils.data.Dataset):
    def __init__(self, dir, kind='train', transform=None):
        assert kind in ['train', 'val']
        assert os.path.isdir(dir)

        self.kind = kind
        self.transform = transform
        self.images_dir = os.path.join(dir, self.kind, 'img')
        self.mask_dir = os.path.join(dir, self.kind, 'label')

        self.file_list = [entry.name for entry in os.scandir(self.images_dir)]
        self.images_list = [os.path.join(self.images_dir, x) for x in self.file_list]
        self.mask_list = [os.path.join(self.mask_dir, x) for x in self.file_list]

        self.colour_map = ColourMap.idx_to_color # tensor of shape (35, 3)
        # self.category_map = ColourMap.name_to_category # dict of shape {index (int): category (int)}
        self.category_map = ColourMap.name_to_category_list


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = io.read_image(self.images_list[idx])
        mask = io.read_image(self.mask_list[idx])

        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        transformed_mask = self.mask_to_class_idx(mask)

        return {
            'image': transformed_image,
            'mask': transformed_mask,
            'original_image': image,
            'original_mask': mask
        }

    def mask_to_class_idx(self, mask):
        # Reshape the mask to (H * W, 3)
        _, H, W = mask.shape
        mask = mask.permute(1, 2, 0).reshape(-1, 3)  # shape: (H * W, 3)

        # Calculate distances between each pixel and every colour in the colour map
        # colour_map shape: (35, 3)
        distances = torch.norm(mask[:, None] - self.colour_map[None, :].float(), dim=2)  # before norm: (H * W, 35, 3), after norm: (H * W, 35)

        # Find the index of the minimum distance (i.e., closest colour)
        mask_idx = torch.argmin(distances, dim=1)  # shape: (H * W)

        # Reshape back to (H, W)
        mask_idx = mask_idx.reshape(H, W)

        # Map the class index to the category index
        mask_idx = self.category_map[mask_idx]

        return mask_idx

# I got it from: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
class ColourMap:
    idx_to_name = [
        "unlabeled",
        "ego vehicle",
        "rectification border",
        "out of roi",
        "static",
        "dynamic",
        "ground",
        "road",
        "sidewalk",
        "parking",
        "rail track",
        "building",
        "wall",
        "fence",
        "guard rail",
        "bridge",
        "tunnel",
        "pole",
        "polegroup",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "caravan",
        "trailer",
        "train",
        "motorcycle",
        "bicycle",
        "license plate",
    ]

    idx_to_category = [
        "void",
        "flat",
        "construction",
        "object",
        "nature",
        "sky",
        "human",
        "vehicle",
    ]

    idx_to_color = torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [111, 74, 0],
        [81, 0, 81],
        [128, 64, 128],
        [244, 35, 232],
        [250, 170, 160],
        [230, 150, 140],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [180, 165, 180],
        [150, 100, 100],
        [150, 120, 90],
        [153, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 0, 90],
        [0, 0, 110],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 142],
    ], dtype=torch.int64)

    idx_to_color = idx_to_color[:, [2, 1, 0]]

    name_to_category = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 2,
        17: 3,
        18: 3,
        19: 3,
        20: 3,
        21: 4,
        22: 4,
        23: 5,
        24: 6,
        25: 6,
        26: 7,
        27: 7,
        28: 7,
        29: 7,
        30: 7,
        31: 7,
        32: 7,
        33: 7,
        34: 7,
    }

    name_to_category_list = torch.tensor([val for val in name_to_category.values()])
