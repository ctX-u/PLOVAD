import torch
from loss import *
from utils import *

from test import test_func
import copy

def train_func(dataloader, test_loader,model,gt,optimizer, criterion, xentropy, clslist,device,steps,best_auc,cfg,logger,best_model_wts,lamda=0):
    B_loss = []
    M_loss = []


    with torch.set_grad_enabled(True):
        model.train()
        for i, (v_input, label, multi_label) in enumerate(dataloader):
            steps+=1
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().to(device)

            #binary label
            label = label.float().to(device)
            # SET WEIGHT FOR POS IN BCELOSS
            if cfg.IFweight:
                weights = torch.where(label == 1.0, cfg.weight_positive, cfg.weight_negative)
                criterion = torch.nn.BCELoss(weight=weights)
            #----------
            
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
            if steps>10 and steps%10==0:
                auc,top1acc,top5acc,mauc,mauc_WOnorm = test_func(test_loader, model, gt, cfg.dataset,clslist,cfg.device)
                #logger.info('[steps:{}]:  | {}:{:.4f} \n top1ACC:{:.4f} top5ACC:{:.4f} mauc:{:.4f} mauc_ab:{:.4f}'.format(steps,cfg.metrics,auc,top1acc,top5acc,mauc,mauc_WOnorm))
                logger.info('[steps:{}]:  |loss1:{:.4f} loss2:{:.4f} {}:{:.4f} \n top1ACC:{:.4f} top5ACC:{:.4f} mauc:{:.4f} mauc_ab:{:.4f}'.format(steps,loss1.item(),loss2.item(),cfg.metrics,auc,top1acc,top5acc,mauc,mauc_WOnorm))
                if auc >= 0.79:
                        torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_'+str(round(auc, 4)).split('.')[1] + '.pkl')
                
                if auc >= best_auc:
                    best_auc = auc
                
                    best_model_wts = copy.deepcopy(model.state_dict())

                    torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_ckpt_best_tmp.pkl')

                    logger.info('save model at steps{}, {}:{:.4f}\n'.format(steps, cfg.metrics,best_auc))
            # if steps>0 and steps%20==0: break



    return sum(B_loss) / len(B_loss), sum(M_loss) / len(M_loss),steps,best_model_wts,best_auc
