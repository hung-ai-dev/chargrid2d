import glob
import os
import os.path as osp

import albumentations as alb
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from utils import read_json

class SegDataset(Dataset):
    def __init__(self, root, size=(1024, 1024), transform=None):
        super().__init__()
        self.root = root
        self.size = size
        self.transform = transform
        self.img_fol = osp.join(root, 'images')
        self.tensor_fol = osp.join(root, 'tensor_input')
        self.semantic_fol = osp.join(root, 'semantic_gt')
        self.obj_fol = osp.join(root, 'obj_gt')
        self.corpus_path = osp.join(root, 'corpus.json')
        self.target_path = osp.join(root, 'target.json')

        self.img_lst = glob.glob(osp.join(self.img_fol,  '*.png'))
        self.tensor_lst = glob.glob(osp.join(self.tensor_fol,  '*.pt'))
        self.semantic_lst = glob.glob(osp.join(self.semantic_fol,  '*.png'))
        self.obj_lst = glob.glob(osp.join(self.obj_fol,  '*.json'))

        self.idx2name = {}
        for idx, path in enumerate(self.tensor_lst):
            name = osp.basename(path).split('.')[0]
            self.idx2name[idx] = name
        self.corpus = read_json(self.corpus_path)
        self.target = read_json(self.target_path)
    
    def __len__(self):
        return len(self.tensor_lst)

    def __transform__(self, tensor):
        pass
    
    def __getitem__(self, idx):
        name = self.idx2name[idx]

        tensor_path = osp.join(self.tensor_fol, name + '.pt')
        semantic_path = osp.join(self.semantic_fol, name + '.png')
        obj_path = osp.join(self.obj_fol, name + '.json')

        tensor = torch.load(tensor_path)
        semantic = Image.open(semantic_path)
        obj = read_json(obj_path)
        
        img = transforms.functional.to_pil_image(tensor)
        
        mask = np.asarray(semantic)

        return tensor
    
if __name__ == "__main__":
    dataset = SegDataset('./data')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for sample in data_loader:
        print(sample)
