import os

from similarity_search import image_similarity, get_images

import pandas as pd

scans_folder = os.path.join('scans','parkhaus_melaten', '32')
vMRTs_folder = os.path.join('vMRTs','parkhaus_melaten_v1')

# def get_images(folder):
#     dirlist = os.listdir(folder)
#     images = [os.path.join(folder,img) for img in dirlist]
#     image_names = [img[:-4] for img in dirlist]

#     return (images, image_names)

vMRTs = get_images(vMRTs_folder)
scans = get_images(scans_folder)


def similarity_matrix(imgs1, imgs2, model_name='efficientnet_b0'):
    df = pd.DataFrame(index=[img.name for img in imgs1], columns=[img.name for img in imgs2])

    for img1 in imgs1:
        for img2 in imgs2:
            sim = image_similarity(img1, img2, model_name=model_name)
            df.loc[img1.name, img2.name] = sim
    
    return df

df = similarity_matrix(scans, vMRTs[0:2], model_name='efficientnet_b1')
print(df)