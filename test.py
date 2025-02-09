import pandas as pd
import matplotlib.pyplot as plt

def mark_labels_on_df(sim_df, label_df):
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

def visualize_results(df, label_df, show=True, save_imgs_to=None):
    for i in range(len(df.columns.to_list())):
        # nach den labels farblich markieren welche tatsächlich similar sind
        colors = ['green' if label == 1 else 'red' for label in label_df.iloc[:,i]]
        # similarity werte jeder spalte
        similarities = df.iloc[:, i]
        # bezeichnung der indices (zeilen)
        indicies = df.index.to_list()

        # nach absteigenden werten sortieren
        sorted_data = sorted(zip(similarities, colors, indicies), reverse=True)
        sorted_similarities, sorted_colors, sorted_indicies = zip(*sorted_data)

        if 'green' not in sorted_colors:
            continue

        plt.scatter(sorted_indicies, sorted_similarities, c=sorted_colors)

        plt.xlabel('scans')
        #  beschriftung x-achse rotieren für lesbarkeit
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('wireframe: ' + df.columns[i])

        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.clf()

df = pd.DataFrame({'A': [1, 2, 1.5], 'B': [4, 3, 6]}, index=['x', 'y', 'z'])

df_label = pd.DataFrame({'A': [0, 0, 0], 'B': [1, 1, 0]}, index=['x', 'y', 'z'])

print(mark_labels_on_df(df, df_label), '\n')
visualize_results(df, df_label, show=True)