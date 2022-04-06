from functools import partial

from numpy import dtype
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']
        self.t_pooling_met = config['t_pooling']
        self.v_pooling_met = config['v_pooling']
        self.fuse_pooling_met = config['fuse_pooling']
        self.vit_depth = config['vit_depth']
        self.vit_heads = config['vit_heads']
        self.vit_dropout = config['vit_dropout']
        self.config_max_sents = config['num_sents']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], 
            patch_size=16, 
            embed_dim=768, 
            depth=self.vit_depth, 
            num_heads=self.vit_heads, 
            mlp_ratio=4, 
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=self.vit_dropout,
        )    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        self.mlp = nn.Sequential(
                  nn.Linear(
                    self.text_encoder.config.hidden_size, 
                    self.text_encoder.config.hidden_size
                  ),
                  nn.Dropout(0.5),
                  nn.ReLU(),
                  nn.LayerNorm(self.text_encoder.config.hidden_size, eps=1e-6),
                  nn.Linear(
                        self.text_encoder.config.hidden_size, 
                        config['num_label']
                    )
                )

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], 
                patch_size=16, 
                embed_dim=768, 
                depth=self.vit_depth, 
                num_heads=self.vit_heads, 
                mlp_ratio=4, 
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop_rate=self.vit_dropout,
            )
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.mlp_m = nn.Sequential(
                    nn.Linear(
                        self.text_encoder.config.hidden_size, 
                        self.text_encoder.config.hidden_size
                    ),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.LayerNorm(self.text_encoder.config.hidden_size, eps=1e-6),
                    nn.Linear(
                        self.text_encoder.config.hidden_size, 
                        config['num_label']
                    )
            )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.mlp,self.mlp_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
         

    def forward(self, image, text, label, device, alpha=0, train=True):
        output_t = self.get_t_feat(text, device, self.text_encoder)
        output_v = self.get_v_feat(image, device, self.visual_encoder)
        output_fuse = self.get_fuse_feat(output_t, output_v, self.text_encoder)
        logit = self.mlp(output_fuse)
        if train:
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    output_t_m = self.get_t_feat(text, device, self.text_encoder_m)
                    output_v_m = self.get_v_feat(image, device, self.visual_encoder_m)
                    output_fuse_m = self.get_fuse_feat(output_t_m, output_v_m, self.text_encoder_m)
                    prediction_m = self.mlp_m(output_fuse_m)

                loss = (1-alpha)*F.cross_entropy(logit, label) - alpha*torch.sum(
                    F.log_softmax(logit, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                loss = F.cross_entropy(logit, label)                
            return logit, loss 
            
        else:
            loss = F.cross_entropy(logit, label)                
            return logit, loss
 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    def get_t_feat(self, inputs, device, encoder):
        bs = len(inputs)
        b = []
        num_max_sent = 0
        for i in range(bs):
            sents = torch.tensor(inputs[i], dtype=torch.long).to(device)[:self.config_max_sents]
            att_mask = torch.ones(sents.shape, dtype=torch.long).to(device)
            # list of #sents * #words * feature_size
            output = encoder(
                sents,
                attention_mask=att_mask,
                return_dict=True,
                mode='text',
                output_hidden_states=True,
            )
            sent_feat = ALBEF.pooling(output, self.t_pooling_met)
            b.append(sent_feat)
            num_max_sent = max(num_max_sent, sent_feat.size(0))
        ret = torch.zeros(
            bs, 
            num_max_sent, 
            self.text_encoder.config.hidden_size, 
            dtype=torch.float
        ).to(device)

        for i in range(bs):
            ret[i][:b[i].size(0)] = b[i]
        return ret
    

    def get_v_feat(self, inputs, device, encoder):
        bs = inputs.size(0)
        b = []
        for i in range(bs):
            # #images * #channels * h * w
            img = inputs[i]
            # #images * #patches * feature_size
            output = encoder(img)
            # #images * feature_size
            img_feat = ALBEF.v_pooling(output, self.v_pooling_met)
            b.append(img_feat)

        ret = torch.stack(b, dim=0).to(device)
        return ret


    def get_fuse_feat(self, output_t, output_v, encoder):
        output_fuse = encoder(
            encoder_embeds = output_t,
            encoder_hidden_states = output_v,
            mode='fusion',
            return_dict = True,
            output_hidden_states=True,
        )
        ret = ALBEF.pooling(output_fuse, self.fuse_pooling_met)
        return ret


    @staticmethod
    def pooling(hidden_states, method):
        if method == 'last_avg':
            return hidden_states[-1].mean(dim=1)
        elif method == 'last_max':
            return hidden_states[-1].max(dim=1).values
        elif method == 'first_last_avg':
            return (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif method == 'first_last_max':
            return (hidden_states[-1] + hidden_states[1]).max(dim=1).values
        elif method == 'cls':
            return hidden_states[-1][:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(method))


    @staticmethod
    def v_pooling(hidden_states, method):
        if method == 'last_avg':
            return hidden_states.mean(dim=0)
        elif method == 'last_max':
            return hidden_states.max(dim=0).values
        elif method == 'cls':
            return hidden_states[:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(method))