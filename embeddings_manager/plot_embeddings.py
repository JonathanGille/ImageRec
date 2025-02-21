import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

def plot_embeddings(embeddings, label=['Unlabelled'], save_to=None, show_plot=True):
    def match_colors_to_label(available_colors, existing_labels):
        label_color_dictionary = {}
        for n in range(len(existing_labels)):
            label_color_dictionary[existing_labels[n]] = available_colors[n]
        return label_color_dictionary

    # sicherstellen das label strings sind
    label = [str(l) for l in label]

    # farben bestimmen die zu jedem label gehören
    existing_labels = []
    for l in label:
        if l not in existing_labels:
            existing_labels.append(l)

    available_colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'yellow', 'pink', 'brown', 'gray', 'cyan', 'magenta']
    color_label_dic = match_colors_to_label(available_colors, existing_labels)
    
    
    pca = PCA(n_components=2)

    try:
        emb2d = pca.fit_transform(embeddings)
    except:
        try:
            embeddings = [emb.numpy()[0] for emb in embeddings]
            emb2d = pca.fit_transform(embeddings)
        except:
            embeddings = [emb.detach().numpy()[0] for emb in embeddings]
            emb2d = pca.fit_transform(embeddings)
    
    # embeddings nach farben(label) sortieren
    sorted_embeddings_2d = {}
    for l in existing_labels:
        sorted_embeddings_2d[l] = []

    # Beispiel für existing_labels 0,1,2,3,4,5,5:
    # sorted_embeddings_2d = {
    #     '0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': []
    # }

    for i in range(len(emb2d)):
        sorted_embeddings_2d[label[i]].append(emb2d[i])
    
    # scattern nach labels -> farben
    for label, embs in sorted_embeddings_2d.items():
        embs = np.array(embs)
        # falls es keine punkte in der liste gibt, wird ein fehler kommen der übersprungen wird, da leere liste unrelevant ist
        try:
            plt.scatter(embs[:, 0], embs[:, 1], c=color_label_dic[label], marker='o')
        except:
            pass
    
    plt.title("projected embedding space")  # Titel setzen
    # plt.legend(['None','Widerlager_West','Widerlager_Ost','Deck','Seitenansicht','Draufsicht'])  # Legende anzeigen
    plt.legend(existing_labels)  # Legende anzeigen

    if save_to != None:
        plt.savefig(save_to, format='png')
    if show_plot:
        plt.show()
    plt.clf()