import torch
from loss import *
from utils import *
from prompt import text_prompt
def train_func(dataloader, model, optimizer, criterion, xentropy, clslist,device,lamda=0):
    B_loss = []
    M_loss = []
   
    with torch.set_grad_enabled(True):
        model.train()
        for i, (v_input, label, multi_label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().to(device)
           
            #binary label
            label = label.float().to(device)
            #category label targets
            multi_label = multi_label.long().to(device)
            # video_labels = torch.unique(multi_label)

                    #logits:[b,seqlen,1] v_feat:[b,seqlen,512] t_feat:[cls_num,512]
            logits, v_feat,t_feat_pre,t_feat_le = model(v_input, seq_len,clslist)
            # for k,v in model.named_parameters():
                
            #     # print(v.grad)
            #     if v.requires_grad:
                   
            #         if 'mlp1' in k:
            #             print(k)
            #             print(v.grad)
            #     # if k == 'embedding.weight':
            #     #     if v.grad != None:
            #     #         print("embedding:",v.grad.sum())

            logit_scale = model.logit_scale.exp()
            
            #implement of MIL align
            v2t_logits_le = MILAlign(v_feat,t_feat_le,logit_scale,label,seq_len,device)
            v2t_logits_pre = MILAlign(v_feat,t_feat_pre,logit_scale,label,seq_len,device)
            #cross entropy of category label
            loss2 =  xentropy(v2t_logits_le, multi_label) 

            #MIL bce
            loss1 = CLAS(logits, label, seq_len, criterion,device)
#
            loss = loss1 + lamda * loss2 
           


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            B_loss.append(loss1.item())
            M_loss.append(loss2.item())

    return sum(B_loss) / len(B_loss), sum(M_loss) / len(M_loss)
