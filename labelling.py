import pandas as pd
import os

scans_folder = os.path.join('scans','parkhaus_melaten')
vMRTs_folder = os.path.join('vMRTs','parkhaus_melaten_v1')


index_vMRT = ['atest_wl_vMRT', 'front_clipped_1', 'front_clipped_2', 'front_clipped_3', 'side_unclipped_1']

# scans_32_label = {
#     'aatest_wl_scan': [1,1,0,1,0],
#     'ANSICHT SEITE SÜD, M 1  100': [0,0,0,0,1],
#     'ANSICHT WIDERLAGER, M 1 50': [1,1,0,1,0],
#     'DETAIL  A , M 1 25': [0,0,0,0,0],
#     'DETAIL  B , M=1 25': [0,0,0,0,0],
#     'DRAUFSICHT AUF .... N': [0,0,0,0,0],
#     'KAPPE AUF DER STÜTZWAND': [0,0,0,0,0],
#     'LÄNGSSCHNITT A-A, M 1 100': [0,0,0,0,1],
#     'Raumfuge (RF) n. RiZ Fug 1 Bild 2': [0,0,0,0,0],
#     'REGELQUERSCHNITT': [0,0,1,0,0],
#     'SICHTFLÄCHENSCHALUNG   ÜBERBAU': [0,0,0,0,0],
#     'Sickerschacht': [0,0,0,0,0],
# }

# df_label_32 = pd.DataFrame(scans_32_label, index=index_vMRT).T

for name in os.listdir(os.path.join(scans_folder,'33')):
    if name.endswith('.jpg') or name.endswith('.png'):
        print(name)

label_33 = {
    'Absteckpunkte .jpg': [0,0,0,0,0],
    'Absteckpunkte _1.jpg': [0,0,0,0,0],
    'In vertragticher Hinsicht einschließlich der Übereinstimmung mit den.jpg': [0,0,0,0,0],
    'ZEICHNUNGEN UND DER BAUAUSFÜHRUNG ÜBEREIN.jpg': [0,0,0,0,0],
}

label_34 = {

}

# for num in os.listdir(scans_folder):
#     print(os.listdir(os.path.join(scans_folder,num)))
#     break