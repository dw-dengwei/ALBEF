import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from apex import amp
import apex


from models.visual import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from utils import multi_image_collact_fn
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def train(
    model, 
    data_loader, 
    optimizer, 
    tokenizer, 
    epoch, 
    warmup_steps, 
    device, 
    scheduler, 
    config, 
    accumulation_steps):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
 
    for i,(images, text, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, label = images.to(device,non_blocking=True), label.to(device,non_blocking=True)
        if config['num_sents'] != 1:
            text_inputs = tokenizer(text, padding='longest', return_tensors="np")
            text_inputs = utils.split_words(text_inputs.input_ids, device)
        else:
            text_inputs = tokenizer(text, padding='longest', return_tensors="pt")
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        prediction, loss = model(images, text_inputs, device=device, label=label, train=True, alpha=alpha)    
        # loss = loss / accumulation_steps 
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()    

        _, pred_class = prediction.max(1)
        accuracy = (label==pred_class).sum() / label.size(0)
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(accuracy.item(), n=label.size(0))
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
feat_s = []
label_s = []

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 500

    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)   
        
        if config['num_sents'] != 1:
            text_inputs = tokenizer(text, padding='longest', return_tensors="np")  
            text_inputs = utils.split_words(text_inputs.input_ids, device)
        else:
            text_inputs = tokenizer(text, padding='longest', return_tensors="pt")

        prediction, loss, feat = model(images, text_inputs, device=device, label=targets, train=False)  
        feat_s.append(feat)
        label_s.append(targets)
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
def main(args, config):
    utils.init_distributed_mode(args)    

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset(args.dataset, config) 
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        datasets,samplers,
        batch_size=[config['batch_size_train']] + [config['batch_size_test']] * 2,
        num_workers=[0] * 3,
        is_trains=[True,False,False], 
        collate_fns=[None] * 3)

    tokenizer = BertTokenizer.from_pretrained('/home/docker/.cache/huggingface/tokenizer')

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

            for key in list(state_dict.keys()):                
                if 'bert' in key:
                    new_key = key.replace('bert.','')
                    state_dict[new_key] = state_dict[key] 
                    del state_dict[key]
                
        try:
            msg = model.load_state_dict(state_dict,strict=False)
            # if 'amp' in checkpoint.keys():
            #     amp.load_state_dict(checkpoint['amp'])
            print('load checkpoint from %s'%args.checkpoint)
        except RuntimeError:
            print('load chechpoints FAILED.')
        # print(msg)

    model = model.to(device)   
    
    model_without_ddp = model
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if args.distributed:
        model = apex.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module    
    
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    accumulation_steps = config['accumulation_steps']
    best = 0
    best_epoch = 0
    
    print("Start training")
    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(
                model, 
                train_loader, 
                optimizer, 
                tokenizer, 
                epoch, 
                warmup_steps, 
                device, 
                lr_scheduler, 
                config,
                accumulation_steps) 
            
        #val_stats = evaluate(model, val_loader, tokenizer, device, config)
        test_stats = evaluate(model, test_loader, tokenizer, device, config)
        
        if args.evaluate:
            break
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    global feat_s
    global label_s
    feat_s = torch.cat(feat_s)
    label_s = torch.cat(label_s)
    print(feat_s.size())
    print(label_s.size())
    #np.save("output/BO_feat.npy", feat_s.data.cpu().numpy())
    #np.save("output/BO_label.npy", label_s.data.cpu().numpy())
    
    #np.save("output/CH_feat.npy", feat_s.data.cpu().numpy())
    #np.save("output/CH_label.npy", label_s.data.cpu().numpy())
    
    #np.save("output/LA_feat.npy", feat_s.data.cpu().numpy())
    #np.save("output/LA_label.npy", label_s.data.cpu().numpy())
    
    #np.save("output/NY_feat.npy", feat_s.data.cpu().numpy())
    #np.save("output/NY_label.npy", label_s.data.cpu().numpy())
    
    # np.save("output/SF_feat.npy", feat_s.data.cpu().numpy())
    # np.save("output/SF_label.npy", label_s.data.cpu().numpy())
    
    np.save("output/AVG_feat.npy", feat_s.data.cpu().numpy())
    np.save("output/AVG_label.npy", label_s.data.cpu().numpy())
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d\n"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--config', default='')
    parser.add_argument('--output_dir', default='')  
    parser.add_argument('--checkpoint', default='save/pretrained.pth')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    args.config = os.path.join('configs', args.dataset + '.yaml')
    args.output_dir = os.path.join('output', args.dataset)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.text_encoder = config['bert_tokenizer']

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
