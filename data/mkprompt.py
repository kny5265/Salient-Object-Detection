import os
import cv2
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def make_points(mask_dir, num_pt, center=True):

    bbox_coords = {}
    pt_coords = {}

    pt_array = np.zeros((len(glob.glob(mask_dir + '/*.png')), num_pt, 2))
    bbox_array = np.zeros((len(glob.glob(mask_dir + '/*.png')), 4))

    progress_bar = tqdm(sorted(glob.glob(mask_dir + '/*.png')))

    for i, f in enumerate(progress_bar):
        fn = os.path.basename(f).split('.')[0]   
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cont = contours[0]

        if len(contours) > 1:
            for j in range(len(contours) - 1):
                cont = np.concatenate((cont, contours[j+1]), axis=0)

        x,y,w,h = cv2.boundingRect(cont)
        height, width, _ = img.shape
        bbox_array[i, :] = np.array([x, y, x + w, y + h])

        mmt = cv2.moments(cont)
        x_p = int(mmt['m10']/mmt['m00'])
        y_p = int(mmt['m01']/mmt['m00'])
        point = np.array([x_p, y_p])
        
        if center:
            a = np.random.choice(cont.shape[0], num_pt-1, replace=False)
            
            for n in range(len(a)):      
                random_pt = np.concatenate((cont[a[n]], np.expand_dims(point, axis=0)), axis=0)
                random_pt = np.mean(random_pt, axis=0)
                pt_array[i, n, :] = random_pt

            pt_array[i, -1, :] = np.array(point)
        else:
            a = np.random.choice(cont.shape[0], num_pt, replace=False)
            
            for n in range(len(a)):      
                random_pt = np.concatenate((cont[a[n]], np.expand_dims(point, axis=0)), axis=0)
                random_pt = np.mean(random_pt, axis=0)
                pt_array[i, n, :] = random_pt

    os.makedirs('/workspace/surgical_dataset/test/prompt/new', exist_ok=True)
    np.save('/workspace/surgical_dataset/test/prompt/new/bbox.npy', bbox_array)
    np.save('/workspace/surgical_dataset/test/prompt/new/pt.npy', pt_array)

if __name__ == "__main__":

    mask_dir = '/workspace/surgical_dataset/test/mask'

    make_points(mask_dir, 4, center=True)