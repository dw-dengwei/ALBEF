import torch
import torch.nn.functional as F
import numpy as np

def split_words(input_ids, device):
    '''
    input_ids: bs * num_max_words 
    '''
    bs = input_ids.shape[0]
    ret = []
    for b_id in range(bs):
        max_num_words = 0
        sent = input_ids[b_id][1:]
        if sent[-1] == 0:
            sent = sent[:(sent==0).nonzero()[0].min()]
        idx = (sent == 102).nonzero()[0] + 1
        sents = np.split(sent, idx[:-1])
        for i in range(len(sents)):
            sents[i] = np.append(101, sents[i])
            if sents[i].size > 500:
                sents[i] = sents[i][:499]
                sents[i] = np.append(sents[i], 102)
            max_num_words = max(max_num_words, sents[i].size)

        for i in range(len(sents)):
            if sents[i].size != max_num_words:
                sents[i] = np.append(sents[i], [0] * (max_num_words - sents[i].size))
        ret.append(torch.tensor(np.array(sents), dtype=torch.long).to(device))
    return ret


@torch.no_grad()
def predict(
    model, 
    tokenizer, 
    device,
    text,
    image,
):
    model.eval()

    image = image.to(device,non_blocking=True)
    
    text_inputs = tokenizer(text, padding='longest', return_tensors="np")  
    text_inputs = split_words(text_inputs.input_ids, device)

    logits, _ = model(image, text_inputs, device=device, label=0, train=False)  
    logits = F.softmax(logits)
    _, pred_class = logits.max(1)

    return pred_class, logits


