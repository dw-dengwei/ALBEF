from models.model import predict
from models.tokenization_bert import BertTokenizer
from flask import Flask, request
from data.preprocess import pre_image, pre_text
import torch
import json


def env():
    device = torch.device('cuda')
    model = None.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, device


app = Flask(__name__)
@app.route('/', methods=['POST'])
def route():
    global model, tokenizer, device
    text = pre_text(request.form.get('text'))
    image = pre_image(request.files['image'].read())
    pred_class, logits = predict(
        model,
        tokenizer,
        device,
        text,
        image,
    ) 

    return json.encoder({
        'pred_class': pred_class,
        'logits': logits
    })


if __name__ == '__main__':
    global model, tokenizer, device
    model, tokenizer, device = env()
    app.run(host='0.0.0.0',port=5000)