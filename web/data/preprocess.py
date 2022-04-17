from io import BytesIO
from PIL import Image
import re


def pre_text(text):
    url_pattern = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    text = re.sub(url_pattern, '', text)
    at_pattern = '@[a-zA-Z0-9]*'
    text = re.sub(at_pattern, '', text)
    not_ascii_pattern = '[^a-zA-Z0-9]'
    text = re.sub(not_ascii_pattern, ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text + '[SEP]'


def pre_image(img_buf, transform):
    img = Image.open(BytesIO(img_buf)).convert('RGB')
    img = transform(img)
    return img
