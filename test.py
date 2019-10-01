from PIL import Image

import cv2
from matplotlib import pyplot as plt
import albumentations as alb
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    path = './data/semantic_gt/A0116_AA024_1.png'
    mask = Image.open(path)
    data_1 = np.asarray(mask)
    print(data_1.shape)
    print(data_1[1000:1002, 1000:1002])
    

    path = 'data/tensor_input/A0116_AA024_1.pt'
    data = torch.load(path)
    print(data.size())
    print(data[1000:1002, 1000:1002])

    img = transforms.functional.to_pil_image(data)
    data_2 = np.asarray(img)
    print(data_2.shape)
    print(data_2[1000:1002, 1000:1002])

    aug = alb.Compose([
        alb.LongestMaxSize(2048),
        alb.PadIfNeeded(2048, 2048, border_mode=cv2.BORDER_CONSTANT)
    ])

    print(data_1[1000:1020, 1000:1020])
    print(data_2[1000:1020, 1000:1020])

    augmented = aug(image = data_2, mask = data_1)
    img = augmented['image']
    mask = augmented['mask']

    print(img.shape)
    print(img[1100:1120, 1100:1120])
    print(mask.shape)
    print(mask[1100:1120, 1100:1120])

    img = Image.fromarray(img)
    mask = Image.fromarray(mask)

    plt.imshow(img)
    plt.show()

    plt.imshow(mask)
    plt.show()