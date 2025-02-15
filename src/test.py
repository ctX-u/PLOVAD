from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve
import numpy as np
import torch
from prompt import text_prompt
from utils import MILAlign
import einops
from torchmetrics.classification import Accuracy, MulticlassAUROC



def cal_false_alarm(gt, preds, threshold=0.5):
    preds = list(preds.cpu().detach().numpy())
    gt = list(gt.cpu().detach().numpy())

    preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far


def test_func(dataloader, model, gt, dataset,cls_list,device):
    
    mauc_metric = MulticlassAUROC(num_classes=len(cls_list), average=None, thresholds=None)
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        abnormal_preds = torch.zeros(0).to(device)
        abnormal_labels = torch.zeros(0).to(device)
        normal_preds = torch.zeros(0).to(device)
        normal_labels = torch.zeros(0).to(device)
        # gt_tmp = torch.tensor(gt.copy()).to(device)
        similarity, targets = torch.zeros(0).to(device), torch.zeros(0).to(device)
        for i, (v_input, label,multi_label) in enumerate(dataloader):
            v_input = v_input.float().to(device)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)

            logits,v_feat,t_feat_pre,t_feat_le = model(v_input, seq_len,cls_list)
            logit_scale = model.logit_scale.exp()
            #align and get the simlarity,multilabels of each batch
            v2t_logits_pre = MILAlign(v_feat,t_feat_pre,logit_scale,label,seq_len,device)
            v2t_logits_le = MILAlign(v_feat,t_feat_le,logit_scale,label,seq_len,device)
            #v2t_logits_vis = MILAlign(v_feat,t_feat_vis,logit_scale,label,seq_len,device)
            v2t_logits = torch.where(v2t_logits_le>v2t_logits_pre,v2t_logits_le,v2t_logits_pre)
            # v2t_logits = v2t_logits_le
        
            sim_batch = v2t_logits.softmax(dim=-1)
            target_batch = multi_label.to(device)
            
            sim_batch = torch.mean(sim_batch,dim=0).unsqueeze(0)
            target_batch = target_batch[0].unsqueeze(0)
            
            similarity = torch.cat([similarity, sim_batch], dim=0)
            targets = torch.cat([targets, target_batch], dim=0)
            batch_mcauc = mauc_metric.update(sim_batch, target_batch)
            # binary logits 
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))

        values,indices = similarity.topk(5)
        top1 = (indices[:, 0] == targets).tolist()
        top5 = ((indices == einops.repeat(targets, 'b -> b k', k=5)).sum(-1)).tolist()
        top1ACC = np.array(top1).sum() / len(top1)
        top5ACC = np.array(top5).sum() / len(top5)
        mc_auc = mauc_metric.compute()
        mauc = torch.nanmean(mc_auc[mc_auc!=0])
        mauc_WOnorm = torch.nanmean(mc_auc[1:]) #delete cls normal
        mauc_metric.reset()

        pred = list(pred.cpu().detach().numpy())
        # print("pred:",pred)
     
        if dataset == 'ucf-crime' or dataset == 'shanghaitech':
            # n_far = cal_false_alarm(normal_labels, normal_preds)
            # print("false alarm :", n_far)
            fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
            roc_auc = auc(fpr, tpr)
            pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
            pr_auc = auc(rec, pre)
            return roc_auc,top1ACC,top5ACC,mauc,mauc_WOnorm
        elif dataset == 'ubnormal':
            fpr, tpr, _ = roc_curve(list(gt), pred)
            roc_auc = auc(fpr, tpr)
            return roc_auc,top1ACC,top5ACC,mauc,mauc_WOnorm
        elif dataset == 'xd-violence':
            pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
            pr_auc = auc(rec, pre)
            return pr_auc,top1ACC,top5ACC,mauc,mauc_WOnorm
        else:
            raise RuntimeError('Invalid dataset.')
