import os
import cv2
import json
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from statistics import mean
from collections import defaultdict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import torch
import torch.nn as nn
from torch.nn.functional import threshold, normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# Define Dataset for Salient Object Detection

class SODdataset(Dataset):
    def __init__(self, img_dir, mask_dir, prompt_dir, img_size, prompt='pt'):
        self.img_fn_list, self.mask_fn_list = self.scan_(img_dir, mask_dir)
        self.prompt_dir = prompt_dir
        self.prompt = prompt
        self.transform = ResizeLongestSide(img_size)

    def scan_(self, img_dir, mask_dir):
        img_fn_list = sorted(glob.glob(img_dir + '/*.png'))
        mask_fn_list = sorted(glob.glob(mask_dir + '/*.png'))

        return img_fn_list, mask_fn_list

    def __len__(self):
        return len(self.img_fn_list)

    def __getitem__(self, idx):
        img_fn = self.img_fn_list[idx]
        mask_fn = self.mask_fn_list[idx]

        img = cv2.imread(img_fn)
        original_image_size = img.shape[:2]
        img = self.transform.apply_image(img)
        img = torch.as_tensor(img).permute(2, 0, 1)

        gt_grayscale = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE) / 255
        mask = (gt_grayscale == 1)
        mask_resized = torch.from_numpy(np.resize(mask, (1, mask.shape[0], mask.shape[1])))
        mask = torch.as_tensor(mask_resized > 0, dtype=torch.float32)

        if self.prompt == 'pt':
            pt_array = np.load('{}/pt.npy'.format(self.prompt_dir))
            prompt = pt_array[idx]
            prompt = self.transform.apply_coords(prompt, original_image_size)
            prompt = torch.as_tensor(prompt, dtype=torch.float)

        elif self.prompt == 'bb':
            bb_array = np.load('{}/bbox.npy'.format(self.prompt_dir))
            prompt = bb_array[idx]
            prompt = self.transform.apply_boxes(prompt, original_image_size)
            prompt = torch.as_tensor(prompt, dtype=torch.float)

        return img, mask, prompt, original_image_size

model_type = 'vit_h'
checkpoint = '/workspace/segment-anything/segment_anything/sam_vit_h_4b8939.pth'
device = 'cuda:0'

sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train()

img_dir = '/workspace/surgical_dataset/cropped/img'
mask_dir = '/workspace/surgical_dataset/cropped/mask'
bbox_dir = '/workspace/surgical_dataset/cropped/prompt/new'
pt_dir = '/workspace/surgical_dataset/cropped/prompt/new'
prompt_type = 'pt'

dataset = SODdataset(img_dir, mask_dir, pt_dir, sam_model.image_encoder.img_size, prompt=prompt_type)
valid_len = int(len(dataset) * 0.1)
train_set, valid_set = random_split(dataset, [len(dataset) - valid_len, valid_len])
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, drop_last=False)

lr = 1e-6
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn = torch.nn.MSELoss()
num_epochs = 30
losses = []
best_loss = 999999999

for epoch in range(num_epochs):
    i_start = time.time()

    epoch_losses = []
    progress_bar = tqdm(train_loader)
    for i, data in enumerate(progress_bar):
        img = data[0].float().to(device)
        mask = data[1].float().to(device)
        prompt = data[2].to(device)
        original_image_size = (1024, 1280)

        with torch.no_grad():

            img = sam_model.preprocess(img)
            input_size = tuple(img.shape[-2:])
            
            image_embedding = sam_model.image_encoder(img)

            if dataset.prompt == 'pt':
                label = torch.ones(1,1)
                label_torch = label.expand([img.size()[0], prompt.size()[1]])
                prompt = (prompt, label_torch)

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=prompt,
                    boxes=None,
                    masks=None
                )
            else:
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=prompt,
                    masks=None
                )

        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        loss = loss_fn(binary_mask, mask)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        b_loss = loss / img.size()[0]
        epoch_losses.append(b_loss.item())
    
    i_fin = time.time()
    losses.append(epoch_losses)

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    print(f'{i_fin - i_start} per 1 epoch')

    if mean(epoch_losses) < best_loss:
        os.makedirs('save_ft_sam', exist_ok=True)
        torch.save(sam_model.state_dict(), 'save_ft_sam/sam_model.pth')
        best_loss = mean(epoch_losses)

mean_losses = [mean(x) for x in losses]

plt.plot(list(range(len(mean_losses))), mean_losses)
plt.title('Mean epoch loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.savefig('save_ft_sam/test.png')