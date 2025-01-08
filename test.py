import pandas as pd

index_vMRT = ['atest_wl_vMRT', 'front_clipped_1', 'front_clipped_2', 'front_clipped_3', 'side_unclipped_1']

scans = {
    'aatest_wl_scan': [1,1,0,1,0],
    'ANSICHT SEITE SÜD, M 1  100': [0,0,0,0,1],
    'ANSICHT WIDERLAGER, M 1 50': [1,1,0,1,0],
    'DETAIL  A , M 1 25': [0,0,0,0,0],
    'DETAIL  B , M=1 25': [0,0,0,0,0],
    'DRAUFSICHT AUF .... N': [0,0,0,0,0],
    'KAPPE AUF DER STÜTZWAND': [0,0,0,0,0],
    'LÄNGSSCHNITT A-A, M 1 100': [0,0,0,0,1],
    'Raumfuge (RF) n. RiZ Fug 1 Bild 2': [0,0,0,0,0],
    'REGELQUERSCHNITT': [0,0,1,0,0],
    'SICHTFLÄCHENSCHALUNG   ÜBERBAU': [0,0,0,0,0],
    'Sickerschacht': [0,0,0,0,0],
}

df_label = pd.DataFrame(scans, index=index_vMRT).T
print(df_label)