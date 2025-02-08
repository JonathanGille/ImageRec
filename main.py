import os
import time
import timm

from similarity_search import image_similarity, get_images, get_embedding
from utils import save_df_to_csv, load_df_from_csv, scatter
from labelling import get_labels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Embedding():
    def __init__(self):
        self.embedding = None
        self.label = None
        self.proj_2d = None
        self.name = None

    def create_embedding(self, img, model_name='convnext_base'):
        self.embedding = get_embedding(img, model_name=model_name).numpy()[0]

    def project_2d():
        pass

wireframe_names = ['front_clipped_1', 'front_clipped_2', 'front_clipped_3', 'side_unclipped_1', 'topview_unclipped']

# indexing for labelled DFs
label_assignment_dictionary = {
    'Complicated': -1,
    'None': 0,
    'Widerlager_West': 1, 
    'Widerlager_Ost': 2,
    'Deck': 3,
    'Seitenansicht': 4,
    'Draufsicht': 5,
}
# label = 1 -> Widerlager_West = front_clipped_1 -> [1,0,0,0,0]

# index_wireframes = {
#     'front_clipped_1': 1, 
#     'front_clipped_2': 3,
#     'front_clipped_3': 2, 
#     'side_unclipped_1': 4, 
#     'topview_unclipped': 5,
# }

# wireframe_names = ['widerlager_west_1', 'widerlager_west_2', 'widerlager_west_3', 'widerlager_west_4']

# index_wireframes = {
#     'widerlager_west_1': 1, 
#     'widerlager_west_2': 1,
#     'widerlager_west_3': 1, 
#     'widerlager_west_4': 1, 
# }

# # all
# wireframe_names = ['front_clipped_1', 'front_clipped_2', 'front_clipped_3', 'side_unclipped_1', 'topview_unclipped','widerlager_west_1', 'widerlager_west_2', 'widerlager_west_3', 'widerlager_west_4','widerlager_ost_1', 'widerlager_ost_2', 'widerlager_ost_3', 'widerlager_ost_4']

# index_wireframes = {
#     'front_clipped_1': 1, 
#     'front_clipped_2': 3,
#     'front_clipped_3': 2, 
#     'side_unclipped_1': 4, 
#     'topview_unclipped': 5,
#     'widerlager_west_1': 1, 
#     'widerlager_west_2': 1,
#     'widerlager_west_3': 1, 
#     'widerlager_west_4': 1,
#     'widerlager_ost_1': 2, 
#     'widerlager_ost_2': 2,
#     'widerlager_ost_3': 2, 
#     'widerlager_ost_4': 2, 
# }

wireframe_names = [
    "qs_w2e_east",
    "qs_w2e_middle",
    "qs_w2e_west",
    "sideview",
    "topview",
    "wl_east_east",
    "wl_east_north",
    "wl_east_south",
    "wl_east_top",
    "wl_east_west",
    "wl_west_east",
    "wl_west_north",
    "wl_west_south",
    "wl_west_top",
    "wl_west_west"
]
index_wireframes = {
    "qs_w2e_east": 2,
    "qs_w2e_middle": 3,
    "qs_w2e_west": 1,
    "sideview": 4,
    "topview": 5,
    "wl_east_east": 2,
    "wl_east_north": 2,
    "wl_east_south": 2,
    "wl_east_top": 2,
    "wl_east_west": 2,
    "wl_west_east": 1,
    "wl_west_north": 1,
    "wl_west_south": 1,
    "wl_west_top": 1,
    "wl_west_west": 1
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


scans_folder = os.path.join('scans','parkhaus_melaten')
wireframes_folder = os.path.join('wireframes','parkhaus_melaten_v2')
wireframes_folder = os.path.join('wireframes','all')

wireframes = get_images(wireframes_folder)
all_scans = []
for i in range(32,49):
    scans = scans + get_images(os.path.join(scans_folder,str(i)))
print(len(scans))

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


def many_models_one_scan():
    for _model in timm_models:
        scan = '32'
        df = similarity_matrix(scans, wireframes, model_name=_model, print_result=True)
        df_dir = os.path.join('results', _model)
        if not os.path.exists(df_dir):
            os.mkdir(df_dir)
        save_path = os.path.join(df_dir, 'df.csv')
        save_df_to_csv(df,save_path, index=True)
        visualize_results(df, get_label_df(scan), save_imgs_to=df_dir, show=False)

def one_model_many_scans():
    for q in range(34,35):
        _model = 'convnext_base'
        label_df = get_label_df(get_labels(str(q))).T

        df = similarity_matrix(scans, wireframes, model_name=_model, print_result=True).T
        df_dir = os.path.join('results','west_ost_vMRT' ,_model, str(q))
        os.makedirs(df_dir, exist_ok=True)
        save_path = os.path.join(df_dir, 'df.csv')
        save_df_to_csv(df,save_path, index=True)
        visualize_results(df, label_df, save_imgs_to=df_dir, show=False)

def visualize_df_from_csv():
    scan = '32'
    print([m for m in timm.list_models() if 'convnext' in m])
    df_dir = os.path.join('results', 'efficientnet_b0')
    df = load_df_from_csv(os.path.join(df_dir, 'df.csv'), index=True)
    visualize_results(df, get_label_df(scan), save_imgs_to=None, show=True)

def plot_embeddings():
    # farben bestimmen die zu jedem label gehören
    colors = {
        '0': 'black', '1': 'red', '2': 'blue', '3': 'green', '4': 'yellow', '5': 'purple', '6': 'orange'
    }
    pca = PCA(n_components=2)

    label_table = index_wireframes
    for i in range(32,49):
        label_table.update(get_labels(str(i)))


    images = wireframes + scans
    emb_objects = []
    embeddings = []
    for im in images:
        try:
            if label_table[im.name] > 0:
                emb = Embedding()
                # embedding für iamge erstellen
                emb.create_embedding(im)
                # durch image-namen das zugehörige label finden
                emb.label = label_table[im.name]
                emb.name = im.name
                emb_objects.append(emb)
                embeddings.append(emb.embedding)
        except:
            print('#### ERROR probably key not found in label table: ', im.name)

    print('Total number of embeddings labelled > 0:', len(embeddings))
    # auf 2d-plane projizieren
    emb2d = pca.fit_transform(embeddings)
    
    # embeddings nach farben(label) sortieren
    sorted_embeddings_2d = {
        '0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': []
    }
    for i, emb_obj in enumerate(emb_objects):
        sorted_embeddings_2d[str(emb_obj.label)].append(emb2d[i])
    
    # scattern nach labels -> farben
    for label, embs in sorted_embeddings_2d.items():
        embs = np.array(embs)
        # falls es keine punkte in der liste gibt, wird ein fehler kommen der übersprungen wird, da leere liste unrelevant ist
        try:
            plt.scatter(embs[:, 0], embs[:, 1], c=colors[label], marker='o')
        except:
            pass
        
    # plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # one_model_many_scans()
    plot_embeddings()
