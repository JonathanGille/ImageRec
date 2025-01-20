import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def add(a, b):
    return a+b

def visualize_textcommdands(text):
    print(text.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r'))

def scatter(y, x=None, x_label='x', y_label='y', sort='', highlight_indices=None):
    # if x == None:
    #     x = list(range(len(y)))
    if highlight_indices != None:
        # Erstelle die Plot-Daten
        x_highlight = [x[i] for i in highlight_indices]
        y_highlight = [y[i] for i in highlight_indices]
        x_normal = [x[i] for i in range(len(x)) if i not in highlight_indices]
        y_normal = [y[i] for i in range(len(y)) if i not in highlight_indices]
        plt.scatter(x_normal, y_normal, color='blue', label='Normale Punkte')
        plt.scatter(x_highlight, y_highlight, color='red', label='Hervorgehobene Punkte')
    else:
        plt.scatter(x, y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

def find_outlier(liste):
    # Konvertiere die Liste in ein numpy Array für einfache Berechnungen
    array = np.array(liste)
    
    # Berechne Q1 und Q3
    Q1 = np.percentile(array, 25)
    Q3 = np.percentile(array, 75)
    
    # Berechne den Interquartilsabstand (IQR)
    IQR = Q3 - Q1
    
    # Bestimme die unteren und oberen Schwellenwerte
    untere_schwelle = Q1 - 1.5 * IQR
    obere_schwelle = Q3 + 1.5 * IQR
    
    # Finde die Indizes der Ausreißer
    outlier_indices = [i for i, x in enumerate(array) if x < untere_schwelle or x > obere_schwelle]
    
    return outlier_indices

def save_df_to_csv(df, path, index=False):
    df.to_csv(path, index=index)
    print('##### -> Saved DF as .csv in: ', path)

def load_df_from_csv(path, index=False):
    if index:
        df = pd.read_csv(path, index_col=0)
    else:    
        df = pd.read_csv(path)
    print('##### -> Loaded DF from: ', path)
    return df

if __name__ == '__main__':
    pass