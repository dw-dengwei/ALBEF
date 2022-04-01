import os
import json
import random
from PIL import Image
from importlib_metadata import entry_points
from torch.utils.data import Dataset
from dataset.utils import pre_zol, pre_zol_no_sep
import jsonlines
import torch
import numpy as np

def _load_annotations(annotations_jsonpath):
    entries = []

    with open(annotations_jsonpath, "r", encoding="utf8") as f:
        for annotation in jsonlines.Reader(f):
            entries.append(
                {
                    "label": annotation["Rating"],
                    "text": annotation["Text"],
                    "photos": annotation["Photos"],
                    "aspect": annotation["Aspect"]
                }
            )
    return entries

class zol_dataset(Dataset):
    def __init__(self, data_root, transform, split, config):
        self.im_root = os.path.join(data_root, 'image')
        datapath = os.path.join(data_root, config[split + '_file'])
        self._entry = _load_annotations(datapath)
        self.transform = transform
        
        self._max_num_img = 1

    def __len__(self):
        return len(self._entry)
    
    def __getitem__(self, index):
        entry = self._entry[index]
        label = int(entry['label']) - 1

        im_s = torch.zeros(self._max_num_img, 3, 384, 384)
        cnt = 0
        try:
            photos = entry['photos']
            for im_id in photos:
                image_path = os.path.join(self.im_root, im_id)
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')   
                    image = self.transform(image)
                    im_s[cnt] = image
                    cnt += 1
                    if cnt == self._max_num_img:
                        break
        except KeyError:
            pass
        text = entry['text']
        # aspect = entry['aspect']
        # text = aspect + '.' + text
        text = pre_zol(text)
        return im_s, text, label