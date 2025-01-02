import os
import time

from similarity_search import image_similarity, get_images

import pandas as pd

scans_folder = os.path.join('scans','parkhaus_melaten', '32')
vMRTs_folder = os.path.join('vMRTs','parkhaus_melaten_v1')

vMRTs = get_images(vMRTs_folder)
scans = get_images(scans_folder)

def similarity_matrix(imgs1, imgs2, model_name='efficientnet_b0', print_result=False):
    start_time = time.time()
    df = pd.DataFrame(index=[img.name for img in imgs1], columns=[img.name for img in imgs2])

    for img1 in imgs1:
        for img2 in imgs2:
            sim = image_similarity(img1, img2, model_name=model_name)
            df.loc[img1.name, img2.name] = sim
    
    if print_result:
        end_time = time.time()
        print('\n(model = '+model_name+')\n', df)
        print('\n     ---> Time:', round(end_time - start_time,2), 'sec.\n')

    return df

df = similarity_matrix(scans, vMRTs[0:2], model_name='regnetx_400mf', print_result=True)

### available models on timm
timm_models = [
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'swin_base_patch4_window7_224',
    'convnext_base',
    'regnetx_400mf',
    'resnet50',
    'vgg19',
]