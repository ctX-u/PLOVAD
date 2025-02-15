
def build_config(dataset):
    cfg = type('', (), {})()
        # base settings
    cfg.feat_dim = 512  
    # cfg.hid_dim = 128
    cfg.dropout_gat = 0.6
    cfg.out_dim = 32
    cfg.alpha = 0.1
    cfg.train_bs = 128
    cfg.workers = 4
    cfg.prefix = 16
    cfg.postfix = 16
    cfg.device = "cuda:1"
    cfg.load_ckpt = False
    cfg.WANDB = False
    cfg.eval_on_cpu = False
    cfg.fixed_prompt = False
    cfg.head_num = 4
    if dataset in ['ucf', 'ucf-crime']:
        cfg.dataset = 'ucf-crime'
        cfg.model_name = 'ucf_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = '../../Dataset/VITfeature/ucf/features/'
        cfg.train_list = 'list/base2novel/ucf-vit-train_base+normal.list'
        cfg.test_list = 'list/ucf/ucf-vit-test.list'
        cfg.gt = 'list/ucf/gt-ucf-vit.npy'
        cfg.token_feat = 'list/ucf/ucf-prompt_1_not16_859.npy'
        # cfg.token_feat = 'list/ucf/ucf-prompt_gpt35_2.npy'
        cfg.clslist = "./list/prompt/ucf_cls.txt"
        cfg.has_feature_input = True
        # base2novel test: only base
        # cfg.test_list = './list/base2novel/ucf-vit-test_base+normal.list'
        # cfg.gt = './list/base2novel/gt-ucf_base.npy'
        # base2novel test: only novel
        # cfg.test_list = './list/base2novel/ucf-vit-test_novel+normal.list'
        # cfg.gt = './list/base2novel/gt-ucf_novel.npy'
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        cfg.temporal = True
        # training settings
        cfg.temp = 0.09
        cfg.lamda = 1
        cfg.seed = 11 #9
        # test settings
        cfg.test_bs = 1
        cfg.ckpt_path = ''
        cfg.ckpt_path = './ckpt/base2novel/ucf__8678.pkl'
        # prompt
        cfg.preprompt = False
        cfg.backbone = 'ViT-B/16'
        cfg.mask_rate = 0.55
        cfg.save_dir = './ckpt/base2novel_ucf/ovvad/'
        cfg.logs_dir = './log_info_base2novel_ucf_tmp.log'
        cfg.max_epoch = 50
        cfg.max_seqlen = 200 #200
        cfg.lr = 1e-3 #1e-3
        cfg.std_init = 0.01
        cfg.head_num = 4 #4
        cfg.cls_hidden = 128
    elif dataset in ['sh', 'shanghaitech']:
        cfg.dataset = 'shanghaitech'
        cfg.model_name = 'sh_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = '../../Dataset/VITfeature/sh/features/'
        cfg.train_list = 'list/base2novel_sh/half/sh-vit-train_base+normal.list'
        cfg.test_list = 'list/base2novel_sh/half/sh-vit-test_flag-single.list'
        cfg.gt = 'list/base2novel_sh/half/gt-sh_flag-single.npy'
        cfg.token_feat = 'list/sh/sh-prompt_12.npy'
        cfg.clslist = "./list/prompt/sh_cls_re2.txt"
        # #fully
        # cfg.test_list = 'list/sh/sh-vit-test_flag.list'
        # cfg.gt = 'list/sh/gt-sh.npy'
        # cfg.token_feat = 'list/sh/sh-prompt_all.npy'
        # cfg.clslist = "./list/prompt/class_sh.txt"
        cfg.has_feature_input = True
        # base2novel test
        # cfg.test_list = 'list/base2novel_sh/half/sh-vit-test_novel+normal.list'
        # cfg.gt = 'list/base2novel_sh/half/gt-sh_novel+normal.npy'

        # cfg.test_list = 'list/base2novel_sh/half/sh-vit-test_base+normal.list'
        # cfg.gt = 'list/base2novel_sh/half/gt-sh_base+normal.npy'
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        cfg.temporal = True
        # training settings
        cfg.temp = 0.09
        cfg.lamda = 1
        cfg.seed = 9 #9
        # test settings
        cfg.test_bs = 1
        cfg.ckpt_path = './ckpt/base2novel_sh/half/sh__9798.pkl'  
        # prompt
        cfg.preprompt = False
        cfg.backbone = 'ViT-B/16'
        cfg.mask_rate = 0.9
        cfg.save_dir = './ckpt/base2novel_sh/half/'
        cfg.logs_dir = './log_info_base2novel_sh_tmp.log'
        cfg.max_epoch = 60
        cfg.max_seqlen = 120
        cfg.lr = 5e-5 #5e-4 
        cfg.head_num = 4 #4
        cfg.std_init = 0.02
        cfg.cls_hidden = 128
    elif dataset in ['ub', 'ubnormal']:
        cfg.dataset = 'ubnormal'
        cfg.model_name = 'ub_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = '../../Dataset/VITfeature/UBnormal/features/'
        cfg.train_list = 'list/ubnormal/ub-vit-train.list'
        cfg.test_list = 'list/ubnormal/ub-vit-test.list'
        cfg.gt = 'list/ubnormal/gt.npy'
        cfg.token_feat = 'list/ubnormal/ubnormal-prompt_1_not16.npy'
        cfg.name2clspath = "list/ubnormal/name2cls.json" 
        cfg.clslist = "./list/prompt/class_ubnormal.txt"
        cfg.has_feature_input = True
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        cfg.temporal = True
        # training settings
        cfg.temp = 0.09
        cfg.lamda = 1 #1
        cfg.seed = 2024 #2024
        # test settings
        cfg.test_bs = 1
        cfg.ckpt_path = './ckpt/base2novel_ub/ovvad/ub__6435.pkl'
        # cfg.ckpt_path = ''
        # prompt
        cfg.preprompt = False
        cfg.backbone = 'ViT-B/16'
        cfg.mask_rate = 0.9
        cfg.save_dir = './ckpt/base2novel_ub/ovvad/'
        cfg.logs_dir = './log_info_ub_tmp0928_seed2024_la1_head4_lr1e-3_lossall_step.log'
        cfg.max_epoch = 200
        cfg.max_seqlen = 450 #450
        cfg.lr = 1e-3 #1e-3
        cfg.std_init = 0.01
        cfg.head_num = 4 #4  
        cfg.cls_hidden = 128
        cfg.fixed_prompt = False
        
    return cfg
