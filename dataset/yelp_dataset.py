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

    def clean(self, text):
        text = text.lower()
        url_pattern = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
        text = re.sub(url_pattern, '', text)
        tag_pattern = '#[a-zA-Z0-9]*'
        text = re.sub(tag_pattern, '', text)
        at_pattern = '@[a-zA-Z0-9]*'
        text = re.sub(at_pattern, '', text)
        not_ascii_pattern = '[^a-zA-Z0-9]'
        text = re.sub(not_ascii_pattern, ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        return text

    def pre(self, text):
        words = nltk.tokenize.word_tokenize(text)
        words = [w for w in words if w not in stopwords.words('english')]
        words = [WordNetLemmatizer().lemmatize(w) for w in words]
        return ' '.join(words)

    def sep(self, text):
        pattern = re.compile('.{100}')
        text = '[SEP]'.join(pattern.findall(text))
        if text[-5:] != '[SEP]':
            return text + '[SEP]'
        else:
            return text

    def preprocess(self, text):
        text = self.clean(text)
        text = self.pre(text)
        text = self.sep(text)
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

        text = pre_yelp(self._entry[index]['text'])
        return im_s, text, label
