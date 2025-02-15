
import time
from test import *


def infer_func(model, dataloader, gt, logger, cfg,clslist):
    st = time.time()
    
    mauc_metric = MulticlassAUROC(num_classes=len(clslist), average=None, thresholds=None)
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(cfg.device)
        abnormal_preds = torch.zeros(0).to(cfg.device)
        abnormal_labels = torch.zeros(0).to(cfg.device)
        normal_preds = torch.zeros(0).to(cfg.device)
        normal_labels = torch.zeros(0).to(cfg.device)
        gt_tmp = torch.tensor(gt.copy()).to(cfg.device)
        similarity, targets = torch.zeros(0).to(cfg.device), torch.zeros(0).to(cfg.device)
        for i, (v_input, label,multi_label) in enumerate(dataloader):
            v_input = v_input.float().to(cfg.device)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)   
            logits,v_feat,t_feat_pre,t_feat_le= model(v_input, seq_len,clslist)
          
            np.save(f"result/norm_ucf.npy",logits.squeeze(0).cpu().detach())
            print(logits)
            logit_scale = model.logit_scale.exp()
            #align and get the simlarity,multilabels of each batch
            #v2t_logits = MILAlign(v_feat,t_feat,logit_scale,label,seq_len,cfg.device)
            v2t_logits_pre = MILAlign(v_feat,t_feat_pre,logit_scale,label,seq_len,cfg.device)
            v2t_logits_le = MILAlign(v_feat,t_feat_le,logit_scale,label,seq_len,cfg.device)
           
            v2t_logits = torch.where(v2t_logits_le>v2t_logits_pre,v2t_logits_le,v2t_logits_pre)

            sim_batch = v2t_logits.softmax(dim=-1)
            target_batch = multi_label.to(cfg.device)
                #for multicrop
            sim_batch = torch.mean(sim_batch,dim=0).unsqueeze(0)
            target_batch = target_batch[0].unsqueeze(0)

            similarity = torch.cat([similarity, sim_batch], dim=0)
            targets = torch.cat([targets, target_batch], dim=0)

            batch_mcauc = mauc_metric.update(sim_batch, target_batch)
            # binary logits 
            logits = torch.mean(logits, 0)

            pred = torch.cat((pred, logits))
                #gt(binary),and repeat16
            labels = gt_tmp[: seq_len[0] * 16]
    
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            else:
                abnormal_labels = torch.cat((abnormal_labels, labels))
                abnormal_preds = torch.cat((abnormal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0] * 16:]
        values,indices = similarity.topk(5)
        exit()
        # np.save("result/sim.npy",similarity.cpu().detach().numpy())
        # np.save("result/cls.npy",clslist)
        # np.save("result/target.npy",targets.cpu().detach().numpy())
        # np.save("result/pred.npy",indices[:,0].cpu().detach().numpy())
        top1 = (indices[:, 0] == targets).tolist()
        top5 = ((indices == einops.repeat(targets, 'b -> b k', k=5)).sum(-1)).tolist()
        top1ACC = np.array(top1).sum() / len(top1)
        top5ACC = np.array(top5).sum() / len(top5)
        mc_auc = mauc_metric.compute()
       
        mauc = torch.nanmean(mc_auc[mc_auc!=0])
        mc_auc_ab = mc_auc[1:]
        mauc_WOnorm = torch.nanmean(mc_auc_ab[mc_auc_ab!=0]) #delete cls normal
        mauc_metric.reset()

       
        # n_far = cal_false_alarm(normal_labels, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
    

        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)

    time_elapsed = time.time() - st
    logger.info('AUC:{:.4f} AP:{:.4f} top1ACC:{:.4f} top5ACC:{:.4f} mauc:{:.4f} mauc_ab:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc,pr_auc, top1ACC,top5ACC,mauc,mauc_WOnorm, time_elapsed // 60, time_elapsed % 60))
