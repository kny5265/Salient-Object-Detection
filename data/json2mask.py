import json
import glob
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt

json_list = glob.glob('dataset/json/*.json')

for i in json_list:
    with open("{}".format(i), "r") as read_file:
        data = json.load(read_file)

    all_file_names=list(data.keys())

    Files_in_directory = glob.glob('dataset/*.png')
            
    for j in range(len(all_file_names)): 
        img_name = data[all_file_names[j]]['filename'].split('.')[:-1][0]
        
        image_name = 'dataset/' + img_name + '.png'
        
        if image_name in Files_in_directory: 
            img = np.asarray(PIL.Image.open(image_name))    
        else:
            continue

        if data[all_file_names[j]]['object'] != {}:
            x = 0
            mask = np.zeros((img.shape[0],img.shape[1]))

            while x < len(data[all_file_names[j]]['object']):
                try: 
                    shape1_x=data[all_file_names[j]]['object'][x]['points']['x']
                    shape1_y=data[all_file_names[j]]['object'][x]['points']['y']
                except:
                    shape1_x=data[all_file_names[j]]['object']['points']['x']
                    shape1_y=data[all_file_names[j]]['object']['points']['y']
            

                shape1_x = list(map(float, shape1_x))
                shape1_y = list(map(float, shape1_y))

                shape1_x = list(map(int, shape1_x))
                shape1_y = list(map(int, shape1_y))

                ab=np.stack((shape1_x, shape1_y), axis=1)
                img2=cv2.drawContours(img, [ab], -1, (255,255,255), -1)
                img3=cv2.drawContours(mask, [ab], -1, 255, -1)
                x+=1

            os.makedirs('dataset/binary_masks', exist_ok=True)
            cv2.imwrite('dataset/binary_masks/{}'.format(img_name) +'.png',mask.astype(np.uint8))