import torch
import numpy as np
import torchvision

default_seg_target_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Lambda(lambda x: torch.tensor(np.array(x)))
])
