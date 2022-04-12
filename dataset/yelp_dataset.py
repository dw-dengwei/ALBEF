import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_yelp
import jsonlines
import torch
import numpy as np
import re, nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def _load_annotations(annotations_jsonpath):
    entries = []

    with open(annotations_jsonpath, "r", encoding="utf8") as f:
        for annotation in jsonlines.Reader(f):
            entries.append(
                {
                    "label": annotation["Rating"],
                    "id": annotation["_id"],
                    "text": annotation["Text"],
                    "photos": annotation["Photos"],
                }
            )
    return entries

class yelp_dataset(Dataset):
    def __init__(self, data_root, transform, split, config):
        self.im_root = os.path.join(data_root, 'image')
        datapath = os.path.join(data_root, config[split + '_file'])
        self._entry = _load_annotations(datapath)
        self.transform = transform
        
        self._max_num_img = config['max_image_num']

    def judge(self, text):
        if len(self.clean(text)) <= 1:
            return 'short'
        elif len(self.clean(text)) >= 200:
            return 'long'
        else:
            return 'ok'

    def split_long(self, text):
        ts = text.split('|||')
        for idx, t in enumerate(ts):
            j = self.judge(t)
            if j == 'long':
                pattern = re.compile('.{100}')
                ts[idx] = '|||'.join(pattern.findall(text))
        text = '|||'.join(ts)
        return text 

    def remove_short(self, text):
        ts = text.split('|||')
        ts_temp = ts[:]
        for idx, t in enumerate(ts):
            j = self.judge(t)
            if j == 'short':
                ts_temp.remove(t)
        text = '|||'.join(ts_temp)
        return text

    def clean(self, text):
        url_pattern = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
        text = re.sub(url_pattern, '', text)
        tag_pattern = '#[a-zA-Z0-9]*'
        text = re.sub(tag_pattern, '', text)
        at_pattern = '@[a-zA-Z0-9]*'
        text = re.sub(at_pattern, '', text)
        not_ascii_pattern = '[^a-zA-Z|]'
        text = re.sub(not_ascii_pattern, ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        return text

    def add_sep(self, text):
        url_pattern = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
        text = re.sub(url_pattern, '', text)
        # tag_pattern = '#[a-zA-Z0-9]*'
        # text = re.sub(tag_pattern, '', text)
        at_pattern = '@[a-zA-Z0-9]*'
        text = re.sub(at_pattern, '', text)
        not_ascii_pattern = '[^a-zA-Z0-9|]'
        text = re.sub(not_ascii_pattern, ' ', text)
        text = text.replace('|||', '[SEP]')
        text = re.sub(' +', ' ', text)
        text = text.strip()
        if text[-5:] != '[SEP]':
            return text + '[SEP]'
        else:
            return text

    def __len__(self):
        return len(self._entry)
    
    def __getitem__(self, index):
        entry = self._entry[index]
        label = int(entry['label']) - 1

        im_s = torch.zeros(self._max_num_img, 3, 384, 384)
        cnt = 0
        try:
            photos = entry['photos']
            for im in photos:
                im_id = im['_id']
                image_path = os.path.join(
                    self.im_root,
                    im_id + '.jpg'
                )
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')   
                    # image = Image.new('RGB', (256, 256), (255, 255, 255))
                    image = self.transform(image)
                    im_s[cnt] = image
                    cnt += 1
                    if cnt == self._max_num_img:
                        break
        except KeyError:
            pass
        
        # 预处理 顺序不能变
        text = self._entry[index]['text']
        # text = self.clean(text)
        # text = self.split_long(text)
        # text = self.remove_short(text)
        text = self.add_sep(text)
        return im_s, text, label
