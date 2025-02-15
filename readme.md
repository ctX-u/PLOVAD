# PLOVAD: Prompting Vision-Language Models for Open Vocabulary Video Anomaly Detection
This repository contains the PyTorch implementation of our paper:  [PLOVAD: Prompting Vision-Language Models for Open Vocabulary Video Anomaly Detection](https://ieeexplore.ieee.org/abstract/document/10836858/)

![framework](./pic/framework.pdf)

---
## Setup
### Dependencies
Please set up the environment by following the `requirements.txt` file.

### Dataset
- Official Dataset Download
The original datasets for [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/), [ShanghaiTech](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection), [XD-Violence](https://roc-ng.github.io/XD-Violence/), and [UBnormal](https://github.com/lilygeorgescu/UBnormal?tab=readme-ov-file) can be obtained from their official sources.

- Extract the CLIP feature
    The extracted CLIP features for the UCF-Crime, ShanghaiTech and XD-Violence datasets can be obtained from [CLIP-TSA](https://github.com/joos2010kj/CLIP-TSA).


    You can also use the CLIP model to extract features by referring to the scripts under `./scripts/feature_extract`.

The following files need to be modified in order to run the code on your own machine:

- Change the file paths to the CLIP features of the datasets above in `src/list/`
- Feel free to change the hyperparameters in `configs_base2novel.py`

### About Text Prompts
The Anomaly-specific Prompts (AP) have been generated, transformed into text embeddings, and placed under `src/list`. 

Remember to place the files you use consistently with the path in `src/configs_base2novel.py`

Example scripts are under `scripts/Prompting/`, including

- Get AP using LLM (example glm4 API)

- Extract the text embedding

## Reproduce 
To reproduce the inference results:
- Change the test list path in `src/configs_base2novel.py`, to all/base/novel test set. The 'All' option is set by default in configs_base2novel.py.

- [Download](https://drive.google.com/drive/folders/1TSvamTo6exlvTJnrFf-Gu6uHOztxhD1c?usp=sharing) and move `ckpt/` to your own path, set the ckpt path in `src/configs_base2novel.py`.


- **Inference**
     ```
    cd src
    python main.py --mode infer --dataset ucf
    ```

##  Train and test 
### Train

Example command:

```
cd src
python main.py --mode train --dataset ucf
```

The `--dataset` option can be `ucf`, `sh`, `xd`, or `ub`, referring to UCF-Crime, ShanghaiTech, XD-Violence, or UBnormal.

### Test

Change the ckpt path and test list in `configs_base2novel.py`.

Example command:

 ```
cd src
python main.py --mode infer --dataset ucf
 ```

The `--dataset` option can be `ucf`, `sh`, `xd`, or `ub`, referring to UCF-Crime, ShanghaiTech, XD-Violence, or UBnormal.

## Acknowledgement

Our code references:
- [XDVioDet](https://github.com/Roc-Ng/XDVioDet)
- [PEL4VAD](https://github.com/yujiangpu20/PEL4VAD?tab=readme-ov-file)

## Citation
If you use this code or find our work helpful, please cite our paper:
```bibtex
@article{xu2025plovad,
  title={PLOVAD: Prompting Vision-Language Models for Open Vocabulary Video Anomaly Detection},
  author={Xu, Chenting and Xu, Ke and Jiang, Xinghao and Sun, Tanfeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```


