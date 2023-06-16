# Finetuning for SAM (PytorchLightning ver)

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import os
import cv2
import json
import time
import glob
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_lightning as pl
import logging
import warnings

from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

class SODdataset(Dataset):
    def __init__(self, img_dir, mask_dir, prompt_dir, img_size, prompt='pt'):
        self.img_fn_list, self.mask_fn_list = self.scan_(img_dir, mask_dir)
        self.prompt_dir = prompt_dir
        self.transform = ResizeLongestSide(img_size)
        self.prompt = prompt

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

        return img, mask, prompt

class pl_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        checkpoint = '/workspace/segment-anything/segment_anything/sam_vit_h_4b8939.pth'
        self.model = sam_model_registry['vit_h'](checkpoint=checkpoint)

        self.criterion = torch.nn.MSELoss()
        self.prompt = 'bb'
        self.TensorF = torch.cuda.FloatTensor
        self.TensorL = torch.cuda.LongTensor

        # train
        self.total = 0
        self.dsc = 0
        self.running_loss = 0
        
        # valid
        self.total_val = 0
        self.dsc_val = 0
        self.running_loss_val = 0

        self.best_loss = 99999999

    def forward(self, image_embedding, sparse_embeddings, dense_embeddings):
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        output = (low_res_masks, iou_predictions)
        return output

    def configure_optimizers(self):
        lr = 1e-6
        wd = 0.1
        optimizer = torch.optim.Adam(self.model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img = train_batch[0]
        mask = train_batch[1]
        prompt = train_batch[2]

        original_image_size = (1024, 1280)

        with torch.no_grad():
            img = self.model.preprocess(img).type(self.TensorF)
            input_size = tuple(img.shape[-2:])
            image_embedding = self.model.image_encoder(img)

            if self.prompt == 'pt':
                label = torch.ones(1,1)
                label_torch = label.expand([img.size()[0], prompt.size()[1]])
                prompt = (prompt, label_torch)

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=prompt,
                    boxes=None,
                    masks=None
                )
            else:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=prompt,
                    masks=None
                )
        
        (low_res_masks, iou_predictions) = self(image_embedding, sparse_embeddings, dense_embeddings)

        upscaled_masks = self.model.postprocess_masks(low_res_masks, input_size, original_image_size)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        loss = self.criterion(binary_mask, mask)
        self.running_loss = loss

        self.zero_grad()
        self.total += img.size()[0]
        
        return loss

    def on_train_epoch_end(self) -> None:
        gathered = self.all_gather(self.running_loss)
        if gathered.dim() != 0:
            loss_train = sum([output for output in gathered])
        else:
            loss_train = gathered

        gathered = self.all_gather(self.total)
        if gathered.dim() != 0:
            total = sum([output for output in gathered])
        else:
            total = gathered

        loss_batch = loss_train / total
        self.log('train_loss', loss_batch)

    def on_train_epoch_start(self) -> None:
        self.total = 0
        self.running_loss = 0

    def validation_step(self, val_batch, batch_idx):
        img = val_batch[0]
        mask = val_batch[1]
        prompt = val_batch[2]

        original_image_size = (1024, 1280)

        with torch.no_grad():
            img = self.model.preprocess(img).type(self.TensorF)
            input_size = tuple(img.shape[-2:])
            image_embedding = self.model.image_encoder(img)

            if self.prompt == 'pt':
                prompt = prompt[:, None, :]
                label = torch.tensor([[1.0]])
                label_torch = label.expand([img.size()[0], 1])
                prompt = (prompt, label_torch)

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=prompt,
                    boxes=None,
                    masks=None
                )
            else:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=prompt,
                    masks=None
                )
        
        (low_res_masks, iou_predictions) = self(image_embedding, sparse_embeddings, dense_embeddings)

        upscaled_masks = self.model.postprocess_masks(low_res_masks, input_size, original_image_size)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        val_loss = self.criterion(binary_mask, mask)
        self.running_loss_val = val_loss

        self.total_val += img.size()[0]

        self.zero_grad()

    def on_validation_epoch_end(self) -> None:
        gathered = self.all_gather(self.running_loss_val)
        if gathered.dim() != 0:
            loss_val = sum([output for output in gathered])
        else:
            loss_val = gathered

        gathered = self.all_gather(self.total_val)
        if gathered.dim() != 0:
            total_val = sum([output for output in gathered])
        else:
            total_val = gathered

        loss_val = loss_val / total_val
        self.log('validation_loss', loss_val)

    def on_validation_epoch_start(self) -> None:
        self.total_val = 0
        self.running_loss_val = 0

    def train_dataloader(self):

        img_dir = '/workspace/surgical_dataset/cropped/img'
        mask_dir = '/workspace/surgical_dataset/cropped/mask'
        bbox_dir = '/workspace/surgical_dataset/cropped/prompt/cen'
        pt_dir = '/workspace/surgical_dataset/cropped/prompt/cen'

        dataset = SODdataset(img_dir, mask_dir, pt_dir, self.model.image_encoder.img_size, prompt=self.prompt)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
        
        return train_loader

    def val_dataloader(self):
        img_dir = '/workspace/surgical_dataset/test/img'
        mask_dir = '/workspace/surgical_dataset/test/mask'
        bbox_dir = '/workspace/surgical_dataset/test/prompt/cen'
        pt_dir = '/workspace/surgical_dataset/test/prompt/cen'

        dataset = SODdataset(img_dir, mask_dir, pt_dir, self.model.image_encoder.img_size, prompt=self.prompt)
        valid_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)

        return valid_loader

os.makedirs('save', exist_ok=True)
model = pl_model()
checkpoint_callback = ModelCheckpoint(
    monitor='validation_loss',
    dirpath='save/',
    filename='SAM_FineTuning-epoch{epoch:02d}-val_loss_{validation_loss:.2f}',
    auto_insert_metric_name=False,
    mode="min"
 )

trainer = Trainer(accelerator="gpu", gpus=1, num_nodes=1, max_epochs= 150,callbacks=checkpoint_callback)

trainer.fit(model)