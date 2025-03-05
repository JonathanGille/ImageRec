import os
import time
import timm

from similarity_search import image_similarity, get_images, get_embedding
from utils import save_df_to_csv, load_df_from_csv, scatter
from labelling import get_labels
from embeddings_manager import plot_embeddings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

class Embedding():
    def __init__(self):
        self.embedding = None
        self.label = None
        self.proj_2d = None
        self.name = None

    def create_embedding(self, img, model_name='convnext_base'):
        self.embedding = get_embedding(img, model_name=model_name).numpy()[0]



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

def get_label_df(label_dic, wireframes):
    matching_dic = {}
    wireframe_names = [wf.name for wf in wireframes]
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

    return pd.DataFrame(matching_dic, index=wireframe_names)

scans_folder = os.path.join('scans','parkhaus_melaten')
wireframes_folder = os.path.join('wireframes','all')

wireframes = get_images(wireframes_folder)
all_scans = []
for i in range(32,49):
    all_scans = all_scans + get_images(os.path.join(scans_folder,str(i)))


def visualize_results(df, label_df, show=True, save_imgs_to=None, skip_all_zeros=False):
    for i in range(len(df.columns.to_list())):
        # nach den labels farblich markieren welche tatsächlich similar sind
        colors = ['green' if label == 1 else 'red' for label in label_df.iloc[:,i]]
        similarities = df.iloc[:, i]
        indicies = df.index.to_list()

        # nach absteigenden werten sortieren
        sorted_data = sorted(zip(similarities, colors, indicies), reverse=True)
        sorted_similarities, sorted_colors, sorted_indicies = zip(*sorted_data)

        if skip_all_zeros:
            if 'green' not in sorted_colors:
                continue

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
    # leere df mit imgs1 (wireframes) in zeilen und imgs2 (scans) in spalten erstellen
    df = pd.DataFrame(index=[img.name for img in imgs1], columns=[img.name for img in imgs2])
    print('    - CALCULATING SIMILARITY MATRIX...')
    for img1 in imgs1:
        for img2 in imgs2:
            # similarity wert für jeden scan x wireframe berechnen und an entsprechender location speichern
            sim = image_similarity(img1, img2, model_name=model_name)
            df.loc[img1.name, img2.name] = round(sim,3)
    
    if print_result:
        end_time = time.time()
        print('\n(model = '+model_name+')\n', df)
        print('\n     ---> Time:', round(end_time - start_time,2), 'sec.\n')

    return df

def mark_labels_on_df(sim_df, label_df):
    # label-df maske über similarity-df legen und 1 Werte mit brackets <> markieren
    marked_df = sim_df.copy()
    indices = marked_df.index.tolist()
    columns = marked_df.columns.tolist()
    for ind in indices:
        for col in columns:
            if label_df.loc[ind, col] == 1:
                marked_df.loc[ind, col] = '<'+str(marked_df.loc[ind, col])+'>'
            else:
                marked_df.loc[ind, col] = str(marked_df.loc[ind, col])
    return marked_df

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

def remove_irrelevant_scans(scans, label_dic, labels_to_remove=[0,-1]):
    reduced_scans = []
    reduced_lable_dic = {}
    for i in range(len(scans)):
        if label_dic[scans[i].name] not in labels_to_remove:
            reduced_scans.append(scans[i])
            reduced_lable_dic[scans[i].name] = label_dic[scans[i].name]
    return reduced_scans, reduced_lable_dic

def one_model_many_scans():
    print('starting one_model_many_scans()...')
    for q in range(41,47):
        print('##### '+str(q)+' #####')
        _model = 'convnext_base'

        # erstellte gelabelte dictionarys laden
        labels_dic = get_labels(str(q))
        scans = get_images(os.path.join(scans_folder,str(q)))

        # aus den erstellten label-dictionary ein mask-dataframe erstellen mit 1 (tatsächlich similar) und 0 (tatsächlich dissimilar)  
        label_df = get_label_df(labels_dic, wireframes)
        print('(Label-Dataframe:)\n', label_df, '\n')

        # similarity matrix für alle (scans)x(wireframes) berechnen
        df = similarity_matrix(wireframes, scans, model_name=_model, print_result=True, label_df=None)
        print('(Marked Similarity Matrix:)\n', mark_labels_on_df(df, label_df), '\n')

        # speicherpfad erstellen
        df_dir = os.path.join('results',_model, str(q))
        os.makedirs(df_dir, exist_ok=True)
        save_path = os.path.join(df_dir, 'df.csv')
        save_df_to_csv(df,save_path, index=True)

        # aus df einen plot für jede spalte erstellen und speichern/anzeigen -> skip_all_zeros=True überspringt die Spalten ohne label != 0
        visualize_results(df, label_df, save_imgs_to=df_dir, show=False, skip_all_zeros=True)
        print('--> plots saved')

def visualize_df_from_csv():
    scan = '32'
    print([m for m in timm.list_models() if 'convnext' in m])
    df_dir = os.path.join('results', 'efficientnet_b0')
    df = load_df_from_csv(os.path.join(df_dir, 'df.csv'), index=True)
    # visualize_results(df, get_label_df(get_labels(scan)), save_imgs_to=None, show=True)

def plot_embeddings_custom():
    # farben bestimmen die zu jedem label gehören
    colors = {
        '0': 'black', '1': 'red', '2': 'blue', '3': 'green', '4': 'yellow', '5': 'purple', '6': 'orange'
    }
    pca = PCA(n_components=2)

    label_table = index_wireframes
    for i in range(32,49):
        label_table.update(get_labels(str(i)))


    images = wireframes + all_scans
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

def assemble_images():
    label_table = index_wireframes
    for i in range(32,49):
        label_table.update(get_labels(str(i)))

    images = wireframes + all_scans
    img_label_dic = {}
    for im in images:
        for i in range(0,6):
            if label_table[im.name] == i:
                folder = 'bridge_sections_labelled'
                label_folder = os.path.join(folder, str(i))
                os.makedirs(label_folder, exist_ok=True)
                im.img.save(os.path.join(label_folder, im.name+'.png'))
                

# chekc if the keys in labelling match the names of the scan images. the ones that are output do not match... 
def check_keys():
    for q in range(41,49):
        print('#### '+str(q)+' ####')
        labels_dic = get_labels(str(q))
        scans = get_images(os.path.join(scans_folder,str(q)))
        scan_names = [scan.name for scan in scans]
        for key,value in labels_dic.items():
            if key not in scan_names:
                print(key)
                
def test_plot_embeddings_func():
    images = wireframes[0:5]
    embeddings = [get_embedding(im) for im in images]
    label = [0,1,2,0,4]
    plot_embeddings(embeddings, label)



if __name__ == '__main__':
    # one_model_many_scans()
    # plot_embeddings_custom()
    # check_keys()
    # test_plot_embeddings_func()
    assemble_images()