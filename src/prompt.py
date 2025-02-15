import numpy as np
import clip
from collections import OrderedDict
import json
import torch

def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id


def text_prompt(dataset='ucf'):
    if dataset=='ucf':
        abnormal_dict = {'Normal':0,'Abuse':1, 'Arrest':2, 'Arson':3, 'Assault':4,
                                'Burglary':5, 'Explosion':6, 'Fighting':7,'RoadAccidents':8,
                                'Robbery':9, 'Shooting':10, 'Shoplifting':11, 'Stealing':12, 'Vandalism':13}

        #cls_list = np.array(list(abnormal_dict.values()))
        
        return abnormal_dict


def text_prompting(dataset='ucf', cls="./list/prompt/ucf_cls.txt",clipbackbone='ViT-B/16', device='cpu'):
  
    numC = {'ucf': 14,'sh':16,'xd':7}
    cls_dict = {}
    # load the CLIP model
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    for paramclip in clipmodel.parameters():
        paramclip.requires_grad = False

    if dataset == 'ucf':
        clslist = list(open(cls)) 
        clslist = [a.strip() for a in clslist]
        for id,a in enumerate(clslist):
                cls_dict[a] = id
        clslist = [a.strip().lower() for a in clslist]
    elif dataset == 'sh':
        clslist = list(open(cls)) 
        clslist = [a.strip() for a in clslist]
        for id,a in enumerate(clslist):
                cls_dict[a] = id
        clslist = [a.strip().lower() for a in clslist]
    elif dataset == 'xd':
        clslist = list(open(cls)) 
        clslist = [a.strip() for a in clslist]
        for id,a in enumerate(clslist):
                cls_dict[a] = id
        clslist = [a.strip().lower() for a in clslist]
    elif dataset == 'ub':
        clslist = list(open(cls)) 
        clslist = [a.strip() for a in clslist]
        for id,a in enumerate(clslist):
                cls_dict[a] = id
        clslist = [a.strip().lower() for a in clslist]

 
    return clslist,cls_dict

if __name__ =='__main__':
     text_prompting()