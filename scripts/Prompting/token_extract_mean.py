# !usr/bin/env python3
# -*- coding:UTF-8 -*-

# @Author: Xandra
# @File:token_extract.py
# @Time:2024-02-28 16:57
"""
 prompts to embeddings
 
 """
import os.path

import clip
import torch
import numpy as np
import json
import argparse

def prompt2vec(json_file,prompt_file,clipbackbone='ViT-B/16'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load CLIP and frozen
    clipmodel,_ = clip.load(clipbackbone,device=device,jit=False)
    for param in clipmodel.parameters():
        param.requires_grad = False
    clip_feat = torch.zeros(0).cuda()
    # convert to token embedding
    with open(json_file,'r',encoding='utf-8') as f:
        json_data = json.load(f)
        for label,prompts in json_data.items():
            promptlist = []
            
            for type, prompt in prompts.items():
                promptlist.append(prompt)
                #promptlist.append(f"a video of {label}, "+prompt) #concat each of  “a video of {action}”, to explicitly denote the represented class.
            tokens = clip.tokenize(promptlist).to(device)
            
            with torch.no_grad():
                embeddings = clipmodel.encode_text(tokens)
                embeddings = torch.mean(embeddings,dim=0)
          
            clip_feat = torch.cat((clip_feat,embeddings.view(1,-1)),dim=0)
            print(f"{embeddings.shape} prompts of {label}")
        print(f"final embedding shape is {clip_feat.shape}")
    if not os.path.exists(os.path.dirname(prompt_file)):
        os.makedirs(os.path.dirname(prompt_file))
    np.save(prompt_file,np.array(clip_feat.detach().cpu().numpy()))
if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(description='PromptExtract')
    parser.add_argument('--dataset', default='ubnormal', help='anomaly video dataset')
    args = parser.parse_args()
    json_file = os.path.join('./prompts', f"prompts_{args.dataset}_all.json")
    prompt_file = os.path.join('./prompt_feature', args.dataset+'-prompt.npy')
    prompt2vec(json_file, prompt_file)
