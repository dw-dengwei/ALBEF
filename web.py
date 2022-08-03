import argparse
import os
from tkinter import N
import ruamel_yaml as yaml
import re
from torchvision import transforms
from flask import Flask, request
import re, nltk

import torch

from models.web import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from PIL import Image
from io import BytesIO

import utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

class preprocess:
    def __init__(self, config) -> None:
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        )
        self._test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),
                interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])  

    def _clean(self, text):
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

    def _pre(self, text):
        words = nltk.tokenize.word_tokenize(text)
        return words

    def _sep(self, text):
        n = 100
        i = n
        while i < len(text):
            text.insert(i, '[SEP]')
            i += (n + 1)
        if text[-1] != '[SEP]':
            text = text + ['[SEP]']
        text = ['[CLS]'] + text
        return ' '.join(text)

    def forward_text(self, text: str) -> str: 
        text = self._clean(text)
        text = self._pre(text)
        text = self._sep(text)
        return text

    def forward_image(self, image: Image.Image) -> Image.Image:
        image = image.convert('RGB')
        return self._test_transform(image).unsqueeze(0)
    

@torch.no_grad()
def predict(model, tokenizer, device, images, text):
    # test
    model.eval()
    images = images.to(device,non_blocking=True).unsqueeze(0)
    text = [text]
    
    text_inputs = tokenizer(text, padding='longest', return_tensors="np")  
    text_inputs = utils.split_words(text_inputs.input_ids, device)

    print(text, images.size(), sep='\t')
    prediction = model(images, text_inputs, device=device, label=None, train=False)  

    _, pred_class = prediction.max(1)

    return pred_class, prediction.tolist()
    

def env(args, config):
    print('Loading...')
    device = torch.device('cpu')
    # print(os.getcwd(), args.checkpoint)
    # tokenizer = BertTokenizer.from_pretrained('/home/docker/.cache/huggingface/tokenizer')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = ALBEF(args.text_encoder, tokenizer, config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu') 
    state_dict = checkpoint['model']
    
    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(
        state_dict['visual_encoder.pos_embed'],
        model.visual_encoder
    )
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('Load Checkpoint Finished!')
    # model = model.to(device)   

    pre = preprocess(config)
    return device, tokenizer, model, pre


app = Flask(__name__)
@app.route('/', methods=['POST'])
def controller():
    global device, tokenizer, model, pre
    text = request.form.get('text').encode('utf-8', 
    errors='ignore').decode('utf-8')
    ori_text = text
    text = pre.forward_text(text)
    img_buf = request.files['image'].read()
    img = Image.open(BytesIO(img_buf))
    img = pre.forward_image(img)

    with torch.no_grad():
        pred, logit = predict(model, tokenizer, device, img, text)
    print({
        "prediction": str(pred.tolist()[0]),
        "probability": logit
    })
    return str(pred.tolist()[0])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--config', default='')
    parser.add_argument('--checkpoint', default='save/pretrained.pth')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    args.config = os.path.join('configs','web.yaml')
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.text_encoder = config['bert_tokenizer']

    global device, tokenizer, model, pre
    device, tokenizer, model, pre = env(args, config)
    print('Environment Prepaired!')
    app.run(host='0.0.0.0', port=22333)
