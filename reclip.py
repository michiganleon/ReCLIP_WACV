import os
import re
import json
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from datasets import get_dataset, accuracy
from models import CLIP_LN_T, CLIP_LN_V, LabelPropagationCluster


parser = argparse.ArgumentParser(description='CLIP Evaluation')
parser.add_argument('--dataset', default='resisc45', help='dataset name')
parser.add_argument('--architecture', default='ViT-L/14', help='architecture name', choices=['RN50','RN101','RN50x4','RN50x16','RN50x64','ViT-L/14','ViT-L/14@336px','ViT-B/32','ViT-B/16'])

# training hpyter parameters
parser.add_argument('--bs', '--batch-size', default=64, type=int, metavar='N', dest='bs')
parser.add_argument('--epoch_max', '--epoch_max', default=100, type=int, metavar='N', dest='epoch_max')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--lrt', default=1e-3, type=float, metavar='LR', help='initial learning rate for text encoder', dest='lrt')
parser.add_argument('--lrv', default=1e-3, type=float, metavar='LR', help='initial learning rate for visual encoder', dest='lrv')

# parser.add_argument('--update_basis', action='store_true')
# parser.add_argument('--mean_per_class', action='store_true')

# label propagation parameters
parser.add_argument('--neighbor_size', default=20, type=int)
parser.add_argument('--alpha', default=0.99, type=float)
parser.add_argument('--cut_dim', default=768, type=int)

# training control
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--monitor', action='store_true')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--log', default='./logs/', type=str, help='log dir, should be something like ./logs/')


# main function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(10)

    # load class names and prompts (provided by CLIP)
    with open("./prompts/clip_prompts", 'r') as filename:
        names_prompts = json.load(filename)
        class_names = names_prompts[args.dataset]["classes"]
        templates = names_prompts[args.dataset]["templates"]

    # load the ReCLIP-V model
    v_model = CLIP_LN_V(class_names=class_names, templates=templates, architecture=args.architecture, learnable_classifier=False)
    if torch.cuda.is_available():
        if args.parallel:
            v_model = DataParallel(v_model) # data parallel 
        v_model.to(device)

    # optimizer for ReCLIP-V visual-encoder layer-norm parameters
    if args.parallel:
        v_optimizer = torch.optim.SGD(v_model.module.learnable_params, args.lrv, weight_decay=args.weight_decay, momentum=0.9)
    else:
        v_optimizer = torch.optim.SGD(v_model.learnable_params, args.lrv, weight_decay=args.weight_decay, momentum=0.9)

    # load the ReCLIP-T model
    t_model = CLIP_LN_T(architecture=args.architecture, templates=templates)
    if torch.cuda.is_available():
        if args.parallel:
            t_model = DataParallel(t_model)
        t_model.to(device)
    
    # optimizer for ReCLIP-T text-encoder layer-norm parameters
    if args.parallel:
        t_optimizer = torch.optim.SGD(t_model.module.learnable_params, args.lrt, weight_decay=args.weight_decay, momentum=0.9)
    else:
        t_optimizer = torch.optim.SGD(t_model.learnable_params, args.lrt, weight_decay=args.weight_decay, momentum=0.9)
    
    # obtain datasets with preprocess function provided by CLIP
    if args.parallel:
        test_dataset = get_dataset(dataset_name=args.dataset, preprocess=t_model.module.preprocess)
    else:
        test_dataset = get_dataset(dataset_name=args.dataset, preprocess=t_model.preprocess)

    # set max epoch
    args.epoch_max = min(args.epoch_max, int(len(test_dataset)/5000+2))

    # label propagation module for ReCLIP-V and ReCLIP-T, initialize with classification weights (text embeddings from class names) from CLIP models
    if args.parallel:
        v_label_propagation = LabelPropagationCluster(v_model.module.classification_weight, len(test_dataset), k=args.neighbor_size, alpha=args.alpha, cut_dim=args.cut_dim)
        t_label_propagation = LabelPropagationCluster(v_model.module.classification_weight, len(test_dataset), k=args.neighbor_size, alpha=args.alpha, cut_dim=args.cut_dim)
    else:
        v_label_propagation = LabelPropagationCluster(v_model.classification_weight, len(test_dataset), k=args.neighbor_size, alpha=args.alpha, cut_dim=args.cut_dim)
        t_label_propagation = LabelPropagationCluster(v_model.classification_weight, len(test_dataset), k=args.neighbor_size, alpha=args.alpha, cut_dim=args.cut_dim)

    # loss function
    criterion = torch.nn.CrossEntropyLoss(reduction = 'none')

    # dataloaders
    dataset_size = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, num_workers=32, drop_last=False, shuffle=False)
    train_loader = DataLoader(test_dataset, batch_size=args.bs, num_workers=32, drop_last=False, shuffle=True)

    # logs
    best_acc = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if(not os.path.isdir(args.log)):
        os.mkdir(args.log) # create folder if not exist
    output_file = open(args.log + args.dataset + timestr, 'a')
    json.dump(args.__dict__, output_file, indent=2) # store args for reference
    output_file.flush()
    args.file = output_file 

    for epoch in range(args.epoch_max):
        # test phase: run forward passs over all test data for evalution for both ReCLIP-T and ReCLIP-V; prepare pseudo labels at the same time
        with torch.no_grad():
            top1t, top1v, top1c, n = 0., 0., 0., 0
            
            # monitor setup
            if(args.monitor):
                pbar = tqdm(test_loader)
            else:
                pbar = test_loader
            
            # evaluation starts
            for images, idx, label in pbar:
                # inputs. abusolute idx of current example is also provided in order to record pseudo labels
                image_input = images.to(device)
                label = label.to(device).view(-1)
                
                # forward pass of ReCLIP-T
                t_logits, t_feature = t_model(image_input, class_names)
                t_acc = accuracy(t_logits, label, topk=(1,))[0]
                
                # forward pass of ReCLIP-V
                v_logits, v_feature = v_model(image_input)
                v_acc = accuracy(v_logits, label, topk=(1,))[0]

                # update the label propagation mododules with visual features collected from ReCLIP-V and ReCLIP-T
                t_label_propagation(t_feature, idx, label)
                v_label_propagation(v_feature, idx, label)
                
                # combined logits for prediction
                c_logits = 0.5 * (t_logits + v_logits)
                c_acc = accuracy(c_logits, label, topk=(1,))[0]

                # summary
                top1t += t_acc
                top1v += v_acc
                top1c += c_acc
                n += len(label)

                # update progress bar
                if(args.monitor):
                    pbar.set_description(f"Epoch = {(epoch):d} Test Accuracy (C/T/V) = {(100*top1c/n):.2f}%, {(100*top1t/n):.2f}%, {(100*top1v/n):.2f}%")
            
            # end of evaluation, collected features from all samples, perform label propagation
            # label propagation function returns the accuracy of pseudo labels generated by ReCLIP-T
            pt_acc = t_label_propagation.perform_label_propagation(clear_cache=True) 
            # label propagation function returns the accuracy of pseudo labels generated by ReCLIP-V, as well as the clustering centriods
            pv_acc, centriods = v_label_propagation.perform_label_propagation(clear_cache=True, cluster_centriod=True)
            
            # updates the ReCLIP-V classification weights with clustering centriods (ReCLIP-V uses cluster centriods for classification)
            if args.parallel:
                v_model.module.classification_weight = centriods.t()
            else:
                v_model.classification_weight = centriods.t()

            # logging: update best acc
            if(100 * top1c / n > best_acc):
                best_acc = 100 * top1c / n
            
            # logging: end of epoch summary
            if(args.monitor):
                print(f"Epoch = {(epoch):d} Best Accuracy = {best_acc:.2f}%, Pseudo Label Accuracy (T/V) = {100 * pt_acc:.2f}%, {100 * pv_acc:.2f}%")
            
            # logging
            args.file.write(f"Epoch = {(epoch):d} Test Accuracy (C/T/V) = {(100*top1c/n):.2f}%, {(100*top1t/n):.2f}%, {(100*top1v/n):.2f}%\n")
            args.file.write(f"Epoch = {(epoch):d} Best Accuracy = {best_acc:.2f}%, Pseudo Label Accuracy (T/V) = {100 * pt_acc:.2f}%, {100 * pv_acc:.2f}%\n")
            args.file.flush()
            

        # training phase: updates ReCLIP-T and ReCLIP-V parameters with pseudo labels
        top1t, top1v, top1c, n = 0., 0., 0., 0
        
        # monitor setup
        if(args.monitor):
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader

        # training starts
        for images, idx, label in pbar:
            # inputs. abusolute idx of current example is also provided in order to lookup pseudo labels
            image_input = images.to(device)
            label = label.to(device).view(-1)
            
            # forward pass of ReCLIP-T
            t_logits, _ = t_model(image_input, class_names)
            
            # forward pass of ReCLIP-V
            v_logits, _ = v_model(image_input)
            
            # get pseudo labels for ReCLIP-T, based on current example idx
            t_pseudo_labels, _ = t_label_propagation.get_pseudo_label(idx)
            t_pseudo_labels = torch.LongTensor(t_pseudo_labels).to(device)

            # get pseudo labels for ReCLIP-V, based on current example idx
            v_pseudo_labels, _ = v_label_propagation.get_pseudo_label(idx)
            v_pseudo_labels = torch.LongTensor(v_pseudo_labels).to(device)
            
            # use commonly agreed pseudo labels for training 
            confidence_map = (v_pseudo_labels == t_pseudo_labels)

            # if there is any commonly agreed labels, otherwise (unlikely) skip the current training
            if(torch.sum(confidence_map) > 0):
                
                # back propagation for ReCLIP-T, only updates the entry where both ReCLIP-T and ReCLIP-V agrees
                t_optimizer.zero_grad()
                t_loss = torch.sum(criterion(t_logits, t_pseudo_labels) * confidence_map) / torch.sum(confidence_map)
                t_loss.backward()
                t_optimizer.step()

                # back propagation for ReCLIP-V, only updates the entry where both ReCLIP-T and ReCLIP-V agrees
                v_optimizer.zero_grad()
                v_loss = torch.mean(criterion(v_logits, v_pseudo_labels) * confidence_map) / torch.sum(confidence_map) 
                v_loss.backward()
                v_optimizer.step()

            # accuracy record
            c_logits = 0.5 * (t_logits + v_logits)
            v_acc = accuracy(v_logits, label, topk=(1,))[0]
            t_acc = accuracy(t_logits, label, topk=(1,))[0]
            c_acc = accuracy(c_logits, label, topk=(1,))[0]

            # summary
            top1t += t_acc
            top1v += v_acc
            top1c += c_acc
            n += len(label)

            # update progress bar
            if(args.monitor):
                pbar.set_description(f"Epoch = {(epoch):d} Training Accuracy (C/T/V) = {(100*top1c/n):.2f}%, {(100*top1t/n):.2f}%, {(100*top1v/n):.2f}%")

        # updates the projection matrix and the classification weight in ReCLIP-T
        # for ReCLIP-V, it uses clustering centriods for classification, therefore it does not require this update
        with torch.no_grad():
            if args.parallel:
                classification_weight_t = t_model.module.encode_text(class_names, full_templates=True)
            else:
                classification_weight_t = t_model.encode_text(class_names, full_templates=True)
            
            t_label_propagation.update_projection(classification_weight_t)
            t_label_propagation.update_centriods(classification_weight_t.t())
            
if __name__ == "__main__":
    main()