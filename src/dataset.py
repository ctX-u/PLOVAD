import torch.utils.data as data
from utils import process_feat
import numpy as np
import os
from prompt import *

class UCFDataset(data.Dataset):
    def __init__(self, cfg, cls_dict = None,transform=None, test_mode=False):
        self.feat_prefix = cfg.feat_prefix
        if test_mode:
            self.list_file = cfg.test_list
        else:
            self.list_file = cfg.train_list
        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.normal_flag = 'Normal'
        self.abnormal_dict = cls_dict
       
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))

    def __getitem__(self, index):
        feat_path = os.path.join(self.feat_prefix, self.list[index].strip('\n'))
        video_idx = self.list[index].strip('\n').split('/')[-1].split('_')[0]
        if self.normal_flag in self.list[index]:
            video_ano = video_idx
            ano_idx = self.abnormal_dict[video_ano]
            label = 0.0
        else:
            video_ano = video_idx[:-3]
            ano_idx = self.abnormal_dict[video_ano]
            label = 1.0

        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        
        if self.tranform is not None:
            v_feat = self.tranform(v_feat)
           
        
        if self.test_mode:
            return v_feat, label,ano_idx  # ano_idx , video_name
        else:
            v_feat = process_feat(v_feat, self.max_seqlen, is_random=False)
            return v_feat, label, ano_idx

    def __len__(self):
        return len(self.list)


class SHDataset(data.Dataset):
    def __init__(self, cfg, cls_dict = None,transform=None, test_mode=False):
        self.feat_prefix = cfg.feat_prefix
        if test_mode:
            self.list_file = cfg.test_list
        else:
            self.list_file = cfg.train_list
        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.normal_flag = 'normal'
        self.abnormal_dict = cls_dict
       
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))

    def __getitem__(self, index):
        feat_path = os.path.join(self.feat_prefix, self.list[index].strip('\n'))
        video_idx = self.list[index].strip('\n').split('/')[-1].split('__')[-1].split('.')[0] 
    
        if video_idx == 'throwing_object':
            video_idx = 'throwing object'
           
        if self.normal_flag in self.list[index]:
            video_ano = video_idx
            ano_idx = self.abnormal_dict[video_ano]
            label = 0.0
        else:
            video_ano = video_idx
            
            ano_idx = self.abnormal_dict[video_ano]
            label = 1.0
        
        feat_path = feat_path.split("__")[0] + '.npy'
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        
        if self.tranform is not None:
            v_feat = self.tranform(v_feat)
           
        
        if self.test_mode:
            return v_feat, label,ano_idx  # ano_idx , video_name
        else:
            v_feat = process_feat(v_feat, self.max_seqlen, is_random=False)
            return v_feat, label, ano_idx

    def __len__(self):
        return len(self.list)


class XDDataset(data.Dataset):
    def __init__(self, cfg, cls_dict = None,transform=None, test_mode=False):
        self.feat_prefix = cfg.feat_prefix
        if test_mode:
            self.list_file = cfg.test_list
         
        else:
            self.list_file = cfg.train_list
         
        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.normal_flag = "label_A"
        self.abnormal_dict = cls_dict #{'normal': 0, 'fighting': 1, 'shooting': 2, 'riot': 3, 'abuse': 4, 'car accident': 5, 'explosion': 6}
        with open(cfg.cls2flag,"r") as f:
            self.cls2flag_dic = json.load(f)
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))

    def __getitem__(self, index):
        feat_path = os.path.join(self.feat_prefix, self.list[index].strip('\n'))
         #../../Dataset/VITfeature/xd/features/test/abnormal/Black.Hawk.Down.2001__#01-42-58_01-43-58_label_G-0-0.npy
   
        #video_idx = self.list[index].strip().split('#')[-1].split("_")[-1].split(".")[0].split("-")[0] #B2
        video_idx = self.list[index].strip().split('#')[-1].split("__")[0].split("_")[-1].split(".")[0].split("-")[0]

        if self.normal_flag in self.list[index]:    
            video_ano = self.cls2flag_dic[video_idx].lower()
            ano_idx = self.abnormal_dict[video_ano]
            label = 0.0
        else:
            video_ano = self.cls2flag_dic[video_idx].lower()
            ano_idx = self.abnormal_dict[video_ano]
            
            label = 1.0
        
        # print(video_ano) #Shooting
        # print(ano_idx) #2

        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        
        if self.tranform is not None:
            v_feat = self.tranform(v_feat)
    
        if self.test_mode:
            #[b,t, 512])
            return v_feat, label,ano_idx  # ano_idx , video_name
        else:
            v_feat = process_feat(v_feat, self.max_seqlen, is_random=False)
           
            return v_feat, label, ano_idx

    def __len__(self):
        return len(self.list)

class UBDataset(data.Dataset):
    def __init__(self, cfg, cls_dict = None,transform=None, test_mode=False):
        self.feat_prefix = cfg.feat_prefix
        if test_mode:
            self.list_file = cfg.test_list
        else:
            self.list_file = cfg.train_list
        self.max_seqlen = cfg.max_seqlen
        self.tranform = transform
        self.test_mode = test_mode
        self.normal_flag = 'normal'
        with open(cfg.name2clspath,"r") as f:
            self.name2clsdict = json.load(f)
        self.abnormal_dict = cls_dict
       
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.list_file))

    def __getitem__(self, index):
        feat_path = os.path.join(self.feat_prefix, self.list[index].strip('\n'))
        name = feat_path.split("/")[-1].split(".")[0]
        flag = feat_path.split("/")[-2]
        

        if flag==self.normal_flag:
            video_ano="normal"
            ano_idx = self.abnormal_dict[video_ano]
            label = 0.0
        else:
            video_ano = self.name2clsdict[name]
            ano_idx = self.abnormal_dict[video_ano]
            label = 1.0

        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        
        if self.tranform is not None:
            v_feat = self.tranform(v_feat)
           
        
        if self.test_mode:
            return v_feat, label,ano_idx  # ano_idx , video_name
        else:
            v_feat = process_feat(v_feat, self.max_seqlen, is_random=False)
            return v_feat, label, ano_idx

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    import argparse
    from configs_base2novel import build_config
    from prompt import *
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description='VAD')
    parser.add_argument('--dataset', default='ub', help='anomaly video dataset')
    parser.add_argument('--mode', default='infer', help='model status: (train or infer)')
    args = parser.parse_args()
    cfg = build_config(args.dataset)
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    clslist,cls_dict = text_prompting(dataset=args.dataset, cls=cfg.clslist,clipbackbone=cfg.backbone, device=device)
    train_data = UBDataset(cfg,cls_dict, test_mode=False)

    train_loader = DataLoader(train_data, batch_size=3, shuffle=False,
                             num_workers=0)
    v_feat, label, ano_idx = next(iter(train_loader))
    print(v_feat.shape)
