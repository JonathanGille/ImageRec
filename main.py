import os
import time
import timm

from similarity_search import image_similarity, get_images
from utils import save_df_to_csv, load_df_from_csv, scatter
from labelling import labels

import pandas as pd
import matplotlib.pyplot as plt

wireframe_names = ['front_clipped_1', 'front_clipped_2', 'front_clipped_3', 'side_unclipped_1', 'topview_unclipped']
# indexing for labelled DFs
# label = {
#     'Complicated': -1,
#     'None': 0,
#     'Widerlager_West': 1, 
#     'Widerlager_Ost': 2,
#     'Deck': 3,
#     'Seitenansicht': 4,
#     'Draufsicht': 5,
# }
# label = 1 -> Widerlager_West = front_clipped_1 -> [1,0,0,0,0]

index_wireframes = {
    'front_clipped_1': 1, 
    'front_clipped_2': 3,
    'front_clipped_3': 2, 
    'side_unclipped_1': 4, 
    'topview_unclipped': 5,
}

def get_label_df(label_dic):
    matching_dic = {}
    for scan, scan_label in label_dic.items():
        n = len(wireframe_names)
        vector = [0] * n
        for wireframe, wf_label in index_wireframes.items():
            if scan_label == wf_label:
                correspondig_wf = wireframe
            else:
                correspondig_wf = 'None'
            for k, wf_name in enumerate(wireframe_names):
                if wf_name == correspondig_wf:
                    vector[k] = 1
        matching_dic[scan] = vector

    return pd.DataFrame(matching_dic, index=wireframe_names).T


scans_folder = os.path.join('scans','parkhaus_melaten', '32')
wireframes_folder = os.path.join('wireframes','parkhaus_melaten_v2')

wireframes = get_images(wireframes_folder)
scans = get_images(scans_folder)

# def label_check(lst, labels):
#     for i in range(lst):
#         if labels[i] == 1:
#             df.loc[img1.name, img2.name] = '<'+str(round(sim,3))+'>'
#         else:
#             df.loc[img1.name, img2.name] = str(round(sim,3))

def visualize_results(df, label_df, show=True, save_imgs_to=None):
    for i in range(len(df.columns.to_list())):
        # nach den labels farblich markieren welche tatsächlich similar sind
        colors = ['green' if label == 1 else 'red' for label in label_df.iloc[:,i]]
        similarities = df.iloc[:, i]
        indicies = df.index.to_list()

        # nach absteigenden werten sortieren
        sorted_data = sorted(zip(similarities, colors, indicies), reverse=True)
        sorted_similarities, sorted_colors, sorted_indicies = zip(*sorted_data)

        plt.scatter(sorted_indicies, sorted_similarities, c=sorted_colors)

        plt.xlabel('scans')
        #  beschriftung x-achse rotieren für lesbarkeit
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('wireframe: ' + df.columns[i])

        plt.tight_layout()

        if save_imgs_to != None:
            plt.savefig(os.path.join(save_imgs_to, df.columns[i]+'.png'), format='png')
        if show:
            plt.show()
        else:
            plt.clf()

def visualize_in_one_plot(df, label_df, show=True, save_imgs_to=None):
    for i in range(len(df.columns.to_list())):
        # nach den labels farblich markieren welche tatsächlich similar sind
        colors = ['g' if label == 1 else 'r' for label in label_df.iloc[:,i]]
        similarities = df.iloc[:, i]
        indicies = df.index.to_list()

        # nach absteigenden werten sortieren
        sorted_data = sorted(zip(similarities, colors, indicies), reverse=True)
        sorted_similarities, sorted_colors, sorted_indicies = zip(*sorted_data)

        # plt.scatter(sorted_indicies, sorted_similarities, c=sorted_colors)
        plt.plot(sorted_indicies, sorted_similarities, marker = '.', markersize = 10, c=sorted_colors)

        plt.xlabel('scans')
        #  beschriftung x-achse rotieren für lesbarkeit
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('wireframe: ' + df.columns[i])

        plt.tight_layout()

        if save_imgs_to != None:
            plt.savefig(os.path.join(save_imgs_to, df.columns[i]+'.png'), format='png')
        if show:
            plt.show()
        else:
            plt.clf()

def similarity_matrix(imgs1, imgs2, model_name='efficientnet_b0', print_result=False, label_df=None):
    start_time = time.time()
    df = pd.DataFrame(index=[img.name for img in imgs1], columns=[img.name for img in imgs2])

    for img1 in imgs1:
        for img2 in imgs2:
            sim = image_similarity(img1, img2, model_name=model_name)
            if label_df is not None:
                if label_df.loc[img1.name, img2.name] == 1:
                    df.loc[img1.name, img2.name] = '<'+str(round(sim,3))+'>'
                else:
                    df.loc[img1.name, img2.name] = str(round(sim,3))        
            else:
                df.loc[img1.name, img2.name] = round(sim,3)
    
    if print_result:
        end_time = time.time()
        print('\n(model = '+model_name+')\n', df)
        print('\n     ---> Time:', round(end_time - start_time,2), 'sec.\n')

    return df

# df = similarity_matrix(scans, wireframes, model_name='convnext_large', print_result=True, label_df=df_label_32)

### available models on timm
timm_models = [
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'swin_base_patch4_window7_224',
    'convnext_base',
    'convnext_tiny',
    'convnext_small',
    'convnext_large',
    'regnetx_040',
    'resnet50',
    'vgg19',
]

# for _model in timm_models:
#     df = similarity_matrix(scans, wireframes, model_name=_model, print_result=True)
#     df_dir = os.path.join('results', _model)
#     if not os.path.exists(df_dir):
#         os.mkdir(df_dir)
#     save_path = os.path.join(df_dir, 'df.csv')
#     save_df_to_csv(df,save_path, index=True)
#     visualize_results(df, df_label_32, save_imgs_to=df_dir, show=False)


for q in range(32,39):
    _model = 'convnext_base'
    label_df = get_label_df(labels(str(q)))

    df = similarity_matrix(scans, wireframes, model_name=_model, print_result=True)
    df_dir = os.path.join('results', _model, str(q))
    os.makedirs(df_dir, exist_ok=True)
    save_path = os.path.join(df_dir, 'df.csv')
    save_df_to_csv(df,save_path, index=True)
    visualize_results(df, label_df, save_imgs_to=df_dir, show=False)


# print([m for m in timm.list_models() if 'convnext' in m])
# df_dir = os.path.join('results', 'efficientnet_b0')
# df = load_df_from_csv(os.path.join(df_dir, 'df.csv'), index=True)
# visualize_results(df, label_df, save_imgs_to=None, show=True)