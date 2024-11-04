'''
    extract CLIP features of each videos(have been extract as frames)
    input: pics of frames
    output: CLIP feature 
'''
import torch
import clip
from PIL import Image
import numpy as np
import os
import glob

root = "root path to the dataset" 
vids = [] #dir paths of each video
saveroot = "root save path of the videos" 

os.makedirs(saveroot,exist_ok=True)
vid_names = [] #video names
for root, dirs, files in os.walk(root, topdown=False):
    for name in dirs:
        vids.append(os.path.join(root, name))
        vid_names.append(name)

#load the model
device = "cuda:1" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

print(f"start processing {root}")
lens = len(vids)
for i in range(lens):
    vid_name = vid_names[i]
   
    #obtain all pics of the video dir
    frames = glob.glob(vids[i]+r"/*.jpg")
    savepath = os.path.join(saveroot,vid_name+".npy")
    #start extract CLIP feature
    with torch.no_grad():
        features = np.zeros((0,512))
        for frame in frames:
            image = preprocess(Image.open(frame)).unsqueeze(0).to(device)
            if device=="cuda:1":
                image_features = model.encode_image(image).detach().cpu()
            else:
                image_features = model.encode_image(image)
            features = np.concatenate((features,image_features),axis=0)
    # save features 
    np.save(savepath,features)
    print(f"{i}/{lens} saved in {savepath}, size: {features.shape}")



        
