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


def similarities(imgs1, imgs2):
    df = pd.DataFrame(index=list2, columns=list1)

    # # Werte berechnen und eintragen
    # for row in list2:
    #     for col in list1:
    #         df.loc[row, col] = calculate_value(row, col)
    for vMRT in vMRTs:
        for scan in scans:
            sim = image_similarity(scan, vMRT)
            df.loc[vMRTs, scans] = sim
            # print('\n     ('+scan.name+' <-> '+vMRT.name+')')
            # print(f"Cosine Similarity: {sim:.4f}")
            print(df)


