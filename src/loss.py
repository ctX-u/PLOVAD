import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def CLAS(logits, label, seq_len, criterion,device='cpu'):  
    logits = logits.squeeze()
    ins_logits = torch.zeros(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        if label[i] == 0:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)

            #tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]), largest=True)
            
        else:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))

    clsloss = criterion(ins_logits, label)
    return clsloss

def CLAS_not1(logits, label, seq_len, criterion,device='cpu'): #for xd
    logits = logits.squeeze()
    ins_logits = torch.zeros(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))

    clsloss = criterion(ins_logits, label)
    return clsloss
