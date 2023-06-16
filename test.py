import os
import cv2
import json
import glob
import numpy as np 
import matplotlib.pyplot as plt

from tqdm import tqdm
from statistics import mean
from collections import defaultdict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import torch
import torch.nn as nn
from torch.nn.functional import threshold, normalize

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# Preprocess data 
img_dir = '/workspace/surgical_dataset/test/img'
mask_dir = '/workspace/surgical_dataset/test/mask'
bbox_array = np.load('/workspace/surgical_dataset/test/prompt/rand/bbox.npy')
pt_array = np.load('/workspace/surgical_dataset/test/prompt/rand/pt.npy')

bbox_coords = {}
pt_coords = {}

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def make_dict(mask_dir):
    for i, f in enumerate(sorted(glob.glob(mask_dir + '/*.png'))):
        fn = os.path.basename(f).split('.')[0]
        bbox_coords[fn] = bbox_array[i]
        pt_coords[fn] = pt_array[i]

    return bbox_coords, pt_coords

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

bbox_coords, pt_coords = make_dict(mask_dir)

ground_truth_masks = {}
for k in bbox_coords.keys():
    gt_grayscale = cv2.imread(mask_dir + '/{}.png'.format(k), cv2.IMREAD_GRAYSCALE) / 255
    ground_truth_masks[k] = (gt_grayscale == 1)

# Prepare Fine Tuning
model_type = 'vit_h'
checkpoint = '/workspace/segment-anything/segment_anything/sam_vit_h_4b8939.pth'
device = 'cuda:0'

sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model_orig.to(device)

checkpoint_cen = '/workspace/save_ft_sam/sam_model_rand.pth'
cen_model = sam_model_registry[model_type](checkpoint=checkpoint_cen)
cen_model.to(device)

checkpoint_bb = '/workspace/save_ft_sam/sam_model_rand.pth'
bb_model = sam_model_registry[model_type](checkpoint=checkpoint_bb)
bb_model.to(device)

# Convert the input images into a format SAM's internal functions expect
transformed_data = defaultdict(dict)
for k in bbox_coords.keys():
  image = cv2.imread(img_dir + '/{}.png'.format(k))  
  transform = ResizeLongestSide(cen_model.image_encoder.img_size)
  input_image = transform.apply_image(image)
  input_image_torch = torch.as_tensor(input_image, device=device)
  transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
  
  input_image = cen_model.preprocess(transformed_image)
  original_image_size = image.shape[:2]
  input_size = tuple(transformed_image.shape[-2:])

  transformed_data[k]['image'] = input_image
  transformed_data[k]['input_size'] = input_size
  transformed_data[k]['original_image_size'] = original_image_size

keys = list(bbox_coords.keys())

for k in keys:

    image = cv2.imread(f'{img_dir}/{k}.png')

    predictor_tuned = SamPredictor(cen_model)
    predictor_original = SamPredictor(sam_model_orig)
    predictor_oribb = SamPredictor(bb_model)

    predictor_tuned.set_image(image)
    predictor_original.set_image(image)
    predictor_oribb.set_image(image)


    input_bbox = bbox_coords[k]
    input_pt = pt_coords[k]
    input_label = np.array([1, 1])

    masks_tuned, _, _ = predictor_tuned.predict(
        point_coords=input_pt,
        box=None,
        multimask_output=False,
        point_labels=input_label
    )

    masks_orig, _, _ = predictor_original.predict(
        point_coords=input_pt,
        box=None,
        multimask_output=False,
        point_labels=input_label
    )

    masks_orig_bb, _, _ = predictor_original.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False
    )

    masks_bb, _, _ = predictor_oribb.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False
    )
    
    _, axs = plt.subplots(3, 2, figsize=(25, 25))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(image)
    show_mask(ground_truth_masks[k], axs[0, 1])
    axs[0, 1].set_title('Ground Truth Mask', fontsize=26)
    axs[0, 1].axis('off')

    axs[1, 0].imshow(image)
    show_mask(masks_bb, axs[1, 0])
    show_box(input_bbox, axs[1, 0])
    axs[1, 0].set_title('Mask with Tuned Model(w bbox)', fontsize=26)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(image)
    show_mask(masks_orig_bb, axs[1, 1])
    show_box(input_bbox, axs[1, 1])
    axs[1, 1].set_title('Mask with Untuned Model', fontsize=26)
    axs[1, 1].axis('off')

    axs[2, 0].imshow(image)
    show_mask(masks_tuned, axs[2, 0])
    show_points(input_pt, input_label, axs[2, 0])
    axs[2, 0].set_title('Mask with Tuned Model(w center point)', fontsize=26)
    axs[2, 0].axis('off')

    axs[2, 1].imshow(image)
    show_mask(masks_orig, axs[2, 1])
    show_points(input_pt, input_label, axs[2, 1])
    axs[2, 1].set_title('Mask with Untuned Model', fontsize=26)
    axs[2, 1].axis('off')

    os.makedirs('/workspace/surgical_dataset/test/result', exist_ok=True)
    plt.savefig('/workspace/surgical_dataset/test/result/{}.png'.format(k))

    # --------- DSC score calculation ---------

    ground_truth_masks[k] = np.resize(ground_truth_masks[k], (1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))

    print('After Fine Tuning(w cen) {} : DSC {}'.format(k, dice(masks_tuned, ground_truth_masks[k])))
    print('Original Model(w cen) {} : DSC {}'.format(k, dice(masks_orig, ground_truth_masks[k])))
    print('After Fine Tuning(w bbox) {} : DSC {}'.format(k, dice(masks_bb, ground_truth_masks[k])))
    print('Original Model(w bbox) {} : DSC {}'.format(k, dice(masks_orig_bb, ground_truth_masks[k])))
    