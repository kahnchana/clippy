import numpy as np
from PIL import Image

voc_label_map = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'dining table',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'potted plant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'television monitor'
}


class SegDataset:
    pass


class PascalDataset(SegDataset):

    CLASSES = (
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "television monitor",
        "bag",
        "bed",
        "bench",
        "book",
        "building",
        "cabinet",
        "ceiling",
        "cloth",
        "computer",
        "cup",
        "door",
        "fence",
        "floor",
        "flower",
        "food",
        "grass",
        "ground",
        "keyboard",
        "light",
        "mountain",
        "mouse",
        "curtain",
        "platform",
        "sign",
        "plate",
        "road",
        "rock",
        "shelves",
        "sidewalk",
        "sky",
        "snow",
        "bedclothes",
        "track",
        "tree",
        "truck",
        "wall",
        "water",
        "window",
        "wood",
    )

    def __init__(self, root, split="train", transform=None):
        self.root = root
        assert split in ["train", "val"], f"invalid split: {split}"
        self.split = split
        anno_path = f"{root}/ImageSets/Segmentation/{split}.txt"
        self.file_list = self.get_file_list(anno_path)
        self.label_remap = voc_label_map
        self.transform = transform

    @staticmethod
    def get_file_list(path):
        with open(path, "r") as fo:
            files = fo.readlines()
        files = [x.strip() for x in files]
        return files

    @staticmethod
    def load_image(path, get_arr=True):
        if get_arr:
            return np.array(Image.open(path))
        else:
            return Image.open(path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item, get_arr=True):
        if isinstance(item, int):
            item = item % len(self.file_list)
            cur_idx = self.file_list[item]
        elif isinstance(item, str):
            cur_idx = item
        else:
            raise NotImplementedError(f"Invalid item dtype: {type(item)}")
        img_path = f"{self.root}/JPEGImages/{cur_idx}.jpg"
        seg_path = f"{self.root}/SegmentationClass/{cur_idx}.png"

        if self.transform is not None:
            img = self.load_image(img_path, get_arr=False)
            img = self.transform(img)
        else:
            img = self.load_image(img_path, get_arr=get_arr)
        seg = self.load_image(seg_path, get_arr=get_arr)
        seg_labels = np.unique(seg)
        seg_labels = list(set(seg_labels) - {0, 255})

        return img, seg, seg_labels, cur_idx


if __name__ == '__main__':
    data_dir = "/Users/kanchana/Downloads/VOCdevkit/VOC2012"
    ds = PascalDataset(data_dir)

    from collections import defaultdict

    l_set = defaultdict(int)
    for i in range(50):
        for j in ds[i][2]:
            l_set[j] += 1
